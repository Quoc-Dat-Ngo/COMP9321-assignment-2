"""
Flight Travel Planner API (COMP9321 Assignment 2, due: 6pm - 29/10/2026)

The following file includes several sections such as:
  * configuration, Swagger setup and namespaces
  * helpers shared by every endpoint (custom auth decorator, matching, pagination)
  * SQLAlchemy models + first-run seeding of the `admin` and `user` accounts (as required by the spec)
  * Requirement Set 1 /auth, /users                 - JWT login, account management (admin only)
  * Requirement Set 2 /imports                      - validated, atomic CSV import (admin only)
  * Requirement Set 3 /airports, /airlines, /routes - read-only browsing with filters and links
  * Requirement Set 4 /travel-plans                 - private plans and their routes (owner only)
  * Requirement Set 5 /export                       - JSON file / PNG timeline of the caller's plans

Key design decisions:
  * Authentication is a stateless JWT access token (`Authorization: Bearer <token>`). The token
    only says *who* the caller is; whether the account still exists and is active is re-checked
    in the database on every request, so deactivating or deleting an account takes effect
    immediately even for tokens issued earlier.
  * Every error is JSON of the form {"message": ...} (plus "errors" for payload validation).
  * Private resources (travel plans) that belong to someone else answer 404, never 403, so a
    client cannot discover which ids exist.
  * Every list endpoint uses the same offset pagination (`page`, `per_page`) and envelope.
"""

import io
import json
import os
from collections import Counter
from datetime import UTC, date, datetime
from enum import StrEnum
from functools import wraps

import bcrypt
import matplotlib.dates as mdates
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, g, request
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt,
    get_jwt_identity,
    verify_jwt_in_request,
)
from flask_restx import Api, Resource, abort, fields, inputs, marshal, reqparse
from flask_sqlalchemy import SQLAlchemy
from loguru import logger
from matplotlib.figure import Figure
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage

load_dotenv()

app = Flask(__name__)
# Allow browser clients on other origins (e.g. upload_test.html opened from disk)
CORS(app)

# ---------------------------------
# Config SQLite database connection
# ---------------------------------
# A relative SQLite path: Flask-SQLAlchemy resolves it inside ./instance/ (instance/app.db).
# The file is created on first run and reused afterwards, so data survives restarts.
# DATABASE_URL lets the tests run against a separate throw-away database, e.g.
#   DATABASE_URL=sqlite:///test.db python z5482360_api.py
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
db = SQLAlchemy(app)

# ------------------------
# Configure JWT_SECRET_KEY
# ------------------------
# Generated once with Python's built-in `secrets` library. Read from the environment (.env) when
# set; the fallback keeps the assignment runnable without any setup. Access tokens use
# flask-jwt-extended's default lifetime of 15 minutes, after which the client logs in again.
app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY", "282601b3f714615e2b6d371aa9f2db42240dd7a104c26b61ca3ddc497f04a61a"
)
# Let flask-jwt-extended errors (missing/expired token) reach its own handlers
# instead of being swallowed by flask-restx as a 500
app.config["PROPAGATE_EXCEPTIONS"] = True
# Keep 404 messages as written (no "did you mean ..." suffix from flask-restx)
app.config["ERROR_404_HELP"] = False
# Tolerate a few seconds of clock jitter: a token used immediately after login was otherwise
# rejected as "not yet valid (iat)" when the system clock stepped backwards (seen on WSL)
app.config["JWT_DECODE_LEEWAY"] = 5
jwt = JWTManager(app)


# Return every token problem as 401 with the same {"message": ...} shape as the rest of the API
@jwt.unauthorized_loader
def missing_token_callback(reason):
    return {"message": f"Unauthorised access, please log in ({reason})"}, 401


@jwt.invalid_token_loader
def invalid_token_callback(reason):
    return {"message": f"Invalid token ({reason})"}, 401


@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return {"message": "Token has expired, please log in again"}, 401


# -----------------------------
# API and Swagger configuration
# -----------------------------
authorizations = {
    "Bearer": {
        "type": "apiKey",
        "in": "header",
        "name": "Authorization",
        "description": "Enter: **Bearer &lt;JWT&gt;** (obtain the token from POST /auth/login)",
    }
}

api = Api(
    app,
    version="1.0",
    title="Flight Travel Planner API",
    description="Browse airports, airlines and routes, and manage private travel plans. "
    "Auth: JWT Bearer Token. Roles: Admin / User.",
    authorizations=authorizations,
    security="Bearer",
    prefix="/api/v1",
    doc="/",
)

# Construct namespace
auth_ns = api.namespace("auth", description="Login")
user_ns = api.namespace("users", description="User management (Admin access only)")
dataset_ns = api.namespace(
    "imports", description="Import CSV files (Admin access only)"
)
airport_ns = api.namespace(
    "airports", description="Browse and search airports (any active account, read-only)"
)
airline_ns = api.namespace(
    "airlines", description="Browse and search airlines (any active account, read-only)"
)
route_ns = api.namespace(
    "routes",
    description="Browse and filter routes, e.g. by airline or origin airport "
    "(any active account, read-only)",
)
plan_ns = api.namespace(
    "travel-plans",
    description="Create and manage your own private travel plans (any active account)",
)
export_ns = api.namespace(
    "export",
    description="Export your own travel plans as a JSON file or a timeline image (any active account)",
)


# ---------------------------------------------
# Helper functions, custom decorators and enums
# ---------------------------------------------
# Hashing and checking user password for security purposes
def hash_password(pw: str):
    password_bytes = pw.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password_bytes, salt)
    return hashed_bytes.decode("utf-8")


def check_password(pw: str, hashed_pw: str):
    password_bytes = pw.encode("utf-8")
    hashed_password_bytes = hashed_pw.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)


class UserRole(StrEnum):
    ADMIN = "Admin"
    USER = "User"


# Custom decorator to check the permission (i.e: Role) and the activation status of request user.
# Order of checks matters for the status codes:
#   no/invalid/expired token -> 401 (handled by flask-jwt-extended's loaders above)
#   token of a deleted account -> 401 (the token is valid but no longer identifies anyone)
#   deactivated account       -> 403 (known caller, but not allowed in)
#   wrong role                -> 403
# The role is read from the token claim: roles never change after an account is created, so the
# claim cannot go stale. Activation *can* change, so it is always read from the database.
def require_roles(*allowed_roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            claims = get_jwt()
            user = db.session.get(User, int(get_jwt_identity()))

            if user is None:
                abort(401, "Unauthorised access, please log in")
            if not user.active:
                abort(403, "This account has been deactivated")
            if claims.get("role") not in allowed_roles:
                abort(403, "Insufficient permission to perform this action")
            # Remember the caller so endpoints can check ownership (e.g. travel plans)
            g.current_user = user
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def contains(column, value):
    """Case-insensitive partial match; % and _ typed by the client are matched literally.

    Without escaping, a client searching for "%" would match every row, because % and _ are
    LIKE wildcards. The backslash is escaped first so it can serve as the escape character.
    """
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return column.ilike(f"%{escaped}%", escape="\\")


def paginate(statement, args, key):
    """Run a select for one page and wrap it in the shared page envelope.

    error_out=False: a page past the end returns an empty list instead of a 404, because the
    collection itself exists and `total`/`pages` tell the client where it ends.
    """
    result = db.paginate(
        statement, page=args["page"], per_page=args["per_page"], error_out=False
    )
    return {
        key: result.items,
        "page": result.page,
        "per_page": result.per_page,
        "total": result.total,
        "pages": result.pages,
    }


# -----------------------------------------------------------
# Entity definition/schema for storing in the sqlite database
# -----------------------------------------------------------
# User
class User(db.Model):
    __tablename__ = "users"
    # Never reuse a deleted user's id, so an old token can't map onto a new account
    __table_args__ = {"sqlite_autoincrement": True}

    # Define each user field/column
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(6), nullable=False, default="User")
    active = db.Column(db.Boolean, nullable=False, default=True)

    # Deleting an account deletes its travel plans (SQLite doesn't enforce FKs, the ORM does)
    travel_plans = db.relationship("TravelPlan", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "active": self.active,
        }

    def password_check(self, request_password):
        return check_password(request_password, self.password)

    @classmethod
    def retrieve_user_by_username(cls, username):
        return cls.query.filter_by(username=username).first()


# Airline / Airport / Route mirror the CSV columns one-to-one, because the files define the
# expected schema. Their ids come from the files (autoincrement=False) so that ids stay stable
# across re-imports and travel plans can keep referring to routes by route_id.
# Note: SQLite does not enforce foreign keys unless PRAGMA foreign_keys is switched on, so the
# ForeignKey declarations below document the relationships; referential integrity is enforced
# by the import validation in Requirement Set 2 instead.
class Airline(db.Model):
    __tablename__ = "airlines"

    airline_id = db.Column(db.Integer, primary_key=True, autoincrement=False)
    name = db.Column(db.Text, nullable=False)
    iata_code = db.Column(db.String(5), unique=True, nullable=False)
    icao_code = db.Column(db.String(5), unique=True, nullable=False)
    country = db.Column(db.Text, nullable=False)


# Airport
class Airport(db.Model):
    __tablename__ = "airports"

    airport_id = db.Column(db.Integer, primary_key=True, autoincrement=False)
    name = db.Column(db.Text, nullable=False)
    city = db.Column(db.Text, nullable=False)
    country = db.Column(db.Text, nullable=False)
    iata_code = db.Column(db.String(5), unique=True, nullable=False)
    icao_code = db.Column(db.String(5), unique=True, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    altitude = db.Column(db.Integer, nullable=False)
    timezone = db.Column(db.Text, nullable=False)


# Route
class Route(db.Model):
    __tablename__ = "routes"

    route_id = db.Column(db.Integer, primary_key=True, autoincrement=False)
    airline_id = db.Column(
        db.Integer, db.ForeignKey("airlines.airline_id"), nullable=False
    )
    airline_iata = db.Column(db.String(5), nullable=False)
    origin_airport_id = db.Column(
        db.Integer, db.ForeignKey("airports.airport_id"), nullable=False
    )
    origin_iata = db.Column(db.String(5), nullable=False)
    destination_airport_id = db.Column(
        db.Integer, db.ForeignKey("airports.airport_id"), nullable=False
    )
    destination_iata = db.Column(db.String(5), nullable=False)
    codeshare = db.Column(db.String(5))
    stops = db.Column(db.Integer, nullable=False)
    equipment = db.Column(db.Text, nullable=False)

    # Read-only links to the related rows (imports write the id columns directly).
    # lazy="joined" loads them in the same query as the route, so a page of 100 routes is one
    # query rather than 1 + 300; viewonly because they are never written through the ORM.
    # airline_iata/origin_iata/destination_iata duplicate the related codes on purpose: they
    # are part of the CSV schema, the import guarantees they match, and RS3 filters on them.
    airline = db.relationship("Airline", lazy="joined", viewonly=True)
    origin = db.relationship(
        "Airport", foreign_keys=[origin_airport_id], lazy="joined", viewonly=True
    )
    destination = db.relationship(
        "Airport", foreign_keys=[destination_airport_id], lazy="joined", viewonly=True
    )


# Travel plan (private to its owner)
class TravelPlan(db.Model):
    __tablename__ = "travel_plans"
    __table_args__ = {"sqlite_autoincrement": True}

    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=False, index=True
    )
    # Not unique: different accounts (or the same one) may reuse a name
    name = db.Column(db.String(100), nullable=False)
    departure_date = db.Column(db.Date, nullable=False)
    return_date = db.Column(db.Date, nullable=False)

    route_links = db.relationship(
        "TravelPlanRoute", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def route_ids(self):
        return sorted(link.route_id for link in self.route_links)

    @property
    def routes(self):
        return sorted(
            (link.route for link in self.route_links), key=lambda r: r.route_id
        )


# Many-to-many link between travel plans and routes. One plan has many routes and one route can
# be in many plans, so the plan's metadata lives once in travel_plans and each (plan, route)
# pair is one row here. The composite primary key makes a route appear at most once per plan.
class TravelPlanRoute(db.Model):
    __tablename__ = "travel_plan_routes"

    plan_id = db.Column(db.Integer, db.ForeignKey("travel_plans.id"), primary_key=True)
    route_id = db.Column(db.Integer, db.ForeignKey("routes.route_id"), primary_key=True)

    # Read-only: the route row (with its airline and airports) for exports
    route = db.relationship("Route", lazy="joined", viewonly=True)


# Initiate SQLite database on first start-up,
# reuse existing database i.e: app.db for subsequent run.
# create_all() only creates tables that are missing (it never drops data), and the seeding is
# skipped once `admin` exists, so restarting the server keeps every account and record.
with app.app_context():
    db.create_all()

    if (
        db.session.execute(
            db.select(User).filter_by(username="admin")
        ).scalar_one_or_none()
        is None
    ):
        # Seed 1 unique admin, with 1 user on start-up (as required by the spec)
        admin = User(
            username="admin", password=hash_password("admin"), role=UserRole.ADMIN
        )
        user = User(username="user", password=hash_password("user"), role=UserRole.USER)

        db.session.add_all([admin, user])
        db.session.commit()
        logger.info("Automatically create 1 admin account and 1 user account.")

# ----------------------------------
# Model Schema for API documentation
# ----------------------------------
# strict=True turns into "additionalProperties: false" in the JSON schema, so with
# @expect(..., validate=True) unknown fields (e.g. a client sending "role": "Admin") are
# rejected with 400 instead of being silently ignored.
user_create_model = user_ns.model(
    "UserCreate",
    {
        "username": fields.String(
            required=True,
            description="Account name to log into the app of this user",
            min_length=3,
            max_length=80,
        ),
        "password": fields.String(
            required=True,
            description="Password to log in",
            min_length=3,
            max_length=72,
        ),
    },
    strict=True,
)

user_update_activation_model = user_ns.model(
    "UserActivationUpdate",
    {
        "active": fields.Boolean(
            description="Target status that admin wants to update", required=True
        )
    },
    strict=True,
)

user_response_model = user_ns.model(
    "UserResponse",
    {
        "id": fields.Integer(description="Unique identifier of this user"),
        "username": fields.String(
            description="Account name to log into the app of this user"
        ),
        "role": fields.String(
            description="Assigned role for this user controls their operations in the app"
        ),
        "active": fields.Boolean(description="Activation status for this user"),
    },
    strict=True,
)


def page_model(ns, name, key, item_model):
    """Offset-based pagination shared by every list endpoint."""
    return ns.model(
        name,
        {
            key: fields.List(
                fields.Nested(item_model), description=f"The {key} on this page"
            ),
            "page": fields.Integer(description="Current page number (starts at 1)"),
            "per_page": fields.Integer(description="Maximum number of items per page"),
            "total": fields.Integer(description="Total number of matching items"),
            "pages": fields.Integer(description="Total number of pages"),
        },
    )


# Query parameters shared by every list endpoint; copy() it to add filters
pagination_parser = reqparse.RequestParser()
pagination_parser.add_argument(
    "page",
    type=inputs.positive,
    default=1,
    location="args",
    help="Page number, starting at 1",
)
pagination_parser.add_argument(
    "per_page",
    type=inputs.int_range(1, 100),
    default=10,
    location="args",
    help="Items per page (1-100)",
)

user_page_model = page_model(user_ns, "UserPage", "users", user_response_model)
user_list_parser = pagination_parser.copy()

auth_login_model = auth_ns.model(
    "LogIn",
    {
        "username": fields.String(
            required=True,
            description="Account name to log into the app of this user",
            min_length=3,
            max_length=80,
        ),
        "password": fields.String(
            required=True,
            description="Password to log in",
            min_length=3,
            max_length=72,
        ),
    },
    strict=True,
)

auth_login_response_model = auth_ns.model(
    "LogInResponse",
    {
        "message": fields.String(description="Success log in message"),
        "accessToken": fields.String(
            description="Access token to be place in request header"
        ),
    },
    strict=True,
)


# ---------------------------------------------------
# CRUD Endpoints for authentication and authorisation
# ---------------------------------------------------
@auth_ns.route("/login")
class Login(Resource):
    @auth_ns.expect(auth_login_model, validate=True)
    @auth_ns.doc(
        description="Exchange username and password for a JWT access token. "
        "Send it as `Authorization: Bearer <token>` on every other request.",
        security=None,
    )
    @auth_ns.marshal_with(auth_login_response_model, code=200)
    @auth_ns.response(200, "Log in successfully")
    @auth_ns.response(401, "Invalid username or password")
    @auth_ns.response(403, "Deactivated account cannot access internal service")
    def post(self):
        user_data = auth_ns.payload

        existing_user = User.retrieve_user_by_username(
            username=user_data.get("username").strip()
        )
        # Unknown username and wrong password give the same message, so the endpoint cannot be
        # used to find out which usernames exist. The activation check comes after the password
        # check for the same reason: only the real owner learns the account is deactivated.
        if existing_user is None:
            auth_ns.abort(401, "Invalid username or password")

        if not existing_user.password_check(user_data.get("password")):
            auth_ns.abort(401, "Invalid username or password")

        if not existing_user.active:
            auth_ns.abort(403, "Deactivated account cannot access internal service")

        # The identity is the numeric id (as a string, which PyJWT requires for "sub"), not the
        # username: ids are never reused, so a token of a deleted account cannot silently start
        # working for a new account created later with the same username.
        access_token = create_access_token(
            identity=str(existing_user.id),
            additional_claims={
                "role": existing_user.role,
            },
        )

        return {
            "message": "Logged in successfully",
            "accessToken": access_token,
        }, 200


# ----------------------------------------------------
# Requirement Set 1: Admin CRUD/RESTful API endpoints
# ----------------------------------------------------
@user_ns.route("")
class UserList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @user_ns.expect(user_create_model, validate=True)
    @user_ns.doc(
        description="Add a new User account to the database.\n\n**Access:** Admin only"
    )
    @user_ns.marshal_with(user_response_model, code=201)
    @user_ns.response(201, "User created successfully")
    @user_ns.response(400, "Invalid request payload")
    @user_ns.response(401, "Unauthorised access, please log in")
    @user_ns.response(403, "Insufficient permission and/or deactivated account")
    @user_ns.response(409, "Duplicate username")
    def post(self):
        """Add a new User account"""
        new_user_data = user_ns.payload

        request_username = new_user_data.get("username").strip()
        request_password = new_user_data.get("password")
        if len(request_username) < 3 or len(request_password) < 3:
            user_ns.abort(
                400, "Username and/or password must be at least 3 characters long"
            )

        if (
            db.session.execute(
                db.select(User).filter_by(username=request_username)
            ).scalar_one_or_none()
            is not None
        ):
            user_ns.abort(409, f"User with username {request_username} already existed")

        new_user = User(
            username=request_username,
            password=hash_password(request_password),
            role=UserRole.USER,
        )
        db.session.add(new_user)
        db.session.commit()

        return new_user, 201

    @user_ns.expect(user_list_parser)
    @user_ns.doc(
        description="Retrieve a page of accounts alongside with their details, ordered by id. "
        "Use `page` and `per_page` to navigate; `total` and `pages` tell the client when to stop. "
        "A page past the end returns an empty `users` list.\n\n**Access:** Admin only"
    )
    @user_ns.marshal_with(user_page_model, code=200)
    @user_ns.response(200, "A page of users")
    @user_ns.response(400, "Invalid page or per_page")
    @user_ns.response(401, "Unauthorised access, please log in")
    @user_ns.response(403, "Insufficient permission and/or deactivated account")
    def get(self):
        """Retrieve a paginated list of users"""
        args = user_list_parser.parse_args()
        return paginate(db.select(User).order_by(User.id), args, "users"), 200


@user_ns.route("/<int:user_id>")
@user_ns.param("user_id", "The user's unique ID")
class UserItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @user_ns.doc(
        description="Retrieve the details of one account.\n\n**Access:** Admin only"
    )
    @user_ns.marshal_with(user_response_model, code=200)
    @user_ns.response(200, "User details")
    @user_ns.response(401, "Unauthorised access, please log in")
    @user_ns.response(403, "Insufficient permission and/or deactivated account")
    @user_ns.response(404, "User not found")
    def get(self, user_id):
        """Retrieve a specific user"""
        target_user = db.session.get(User, user_id)
        if target_user is None:
            user_ns.abort(404, f"User {user_id} not found")

        return target_user, 200

    @user_ns.doc(
        description="Delete a user with specified ID, together with all of their travel plans.\n\n**Access:** Admin only"
    )
    @user_ns.response(204, "Deleted this user")
    @user_ns.response(401, "Unauthorised access, please log in")
    @user_ns.response(403, "Insufficient permission to perform this action")
    @user_ns.response(404, "User account not found")
    def delete(self, user_id):
        """Delete a specific user"""
        target_user = db.session.get(User, user_id)
        if target_user is None:
            user_ns.abort(404, "Target user to be deleted not found")

        if target_user.role == "Admin":
            user_ns.abort(403, "Cannot delete Admin account")

        db.session.delete(target_user)
        db.session.commit()

        return "", 204

    @user_ns.expect(user_update_activation_model, validate=True)
    @user_ns.doc(
        description="Activate/Deactivate a User account with a specified user_id.\n\n**Access:** Admin only"
    )
    @user_ns.marshal_with(user_response_model, code=200)
    @user_ns.response(200, "Updated successfully")
    @user_ns.response(401, "Unauthorised access, please log in")
    @user_ns.response(403, "Insufficient permission to perform this action")
    @user_ns.response(404, "User not found")
    def patch(self, user_id):
        """Update user activation status"""
        data = user_ns.payload

        target_status = data.get("active")

        target_user = db.session.get(User, user_id)
        if target_user is None:
            user_ns.abort(404, "Target user to be updated not found")

        if target_user.role == UserRole.ADMIN:
            user_ns.abort(403, "Cannot take action towards Admin account")

        # Setting the status it already has so that PATCH adheres idempotentency
        target_user.active = target_status
        db.session.add(target_user)
        db.session.commit()

        return target_user, 200


# -------------------------------------------------------------
# Requirement Set 2: import airports / airlines / routes (CSV)
# -------------------------------------------------------------
MODELS = {"airlines": Airline, "airports": Airport, "routes": Route}

# Expected schema of each file, taken from the provided samples:
#   column -> (type, required). Column order is not enforced, only the set of names.
SCHEMAS = {
    "airlines": {
        "airline_id": ("int", True),
        "name": ("str", True),
        "iata_code": ("str", True),
        "icao_code": ("str", True),
        "country": ("str", True),
    },
    "airports": {
        "airport_id": ("int", True),
        "name": ("str", True),
        "city": ("str", True),
        "country": ("str", True),
        "iata_code": ("str", True),
        "icao_code": ("str", True),
        "latitude": ("float", True),
        "longitude": ("float", True),
        "altitude": ("int", True),
        "timezone": ("str", True),
    },
    "routes": {
        "route_id": ("int", True),
        "airline_id": ("int", True),
        "airline_iata": ("str", True),
        "origin_airport_id": ("int", True),
        "origin_iata": ("str", True),
        "destination_airport_id": ("int", True),
        "destination_iata": ("str", True),
        "codeshare": ("str", False),  # can be "Y" or empty
        "stops": ("int", True),
        "equipment": ("str", True),
    },
}

# Columns that must hold unique values inside one file
UNIQUE = {
    "airlines": ["airline_id", "iata_code", "icao_code"],
    "airports": ["airport_id", "iata_code", "icao_code"],
    "routes": ["route_id"],
}

# How a route refers to other datasets. Each tuple reads:
#   (route id column, route code column, referenced dataset, its id column, its code column)
# e.g. a route's origin_airport_id must exist in airports.airport_id, and origin_iata must equal
# that airport's iata_code.
ROUTE_LINKS = [
    ("airline_id", "airline_iata", "airlines", "airline_id", "iata_code"),
    ("origin_airport_id", "origin_iata", "airports", "airport_id", "iata_code"),
    (
        "destination_airport_id",
        "destination_iata",
        "airports",
        "airport_id",
        "iata_code",
    ),
]

# Reuse last assignment (i.e: assignment 1)
RANGES = {"latitude": (-90, 90), "longitude": (-180, 180), "stops": (0, 99)}

# Types are checked with regexes on the raw text rather than by letting pandas guess them, so a
# bad value produces a precise message ("'altitude' must be int, found 'high' at line 7")
# instead of a whole column silently becoming text or NaN.
INT_REGEX = r"[+-]?\d+"
FLOAT_REGEX = r"[+-]?\d+(\.\d+)?"

# Request parser for POST /upload
upload_parser = reqparse.RequestParser()
for _name in MODELS:
    upload_parser.add_argument(
        _name,
        location="files",
        type=FileStorage,
        required=False,
        help=f"Optional. The file must be named exactly {_name}.csv",
    )

imported_counts_model = dataset_ns.model(
    "ImportedCounts",
    {
        name: fields.Integer(
            description=f"Number of {name} now stored (omitted if {name}.csv was not uploaded)"
        )
        for name in MODELS
    },
)

import_response_model = dataset_ns.model(
    "ImportResponse",
    {
        "message": fields.String(description="Outcome of the import"),
        "imported": fields.Nested(
            imported_counts_model,
            skip_none=True,
            description="Record count per dataset replaced by this request",
        ),
    },
)


class ImportValidationError(Exception):
    """Raised for any problem with an upload; carries the HTTP status to return."""

    def __init__(self, message, status=400, **details):
        super().__init__(message)
        self.message = message
        self.status = status
        # Extra keys added to the JSON error body
        self.details = details


def dataset_name_matcher(file):
    """Work out which dataset an upload is from its file name only."""
    file_name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]

    for dataset in MODELS:
        if file_name == f"{dataset}.csv":
            return dataset

    raise ImportValidationError(
        f"Unexpected file name '{file.filename}'. "
        f"Upload files named exactly: {', '.join(m + '.csv' for m in MODELS)}"
    )


def parse_and_validate_uploaded_file(name, file):
    """Parse one uploaded CSV into a typed DataFrame, or raise ImportValidationError."""
    schema = SCHEMAS[name]
    if not (file.filename or "").lower().endswith(".csv"):
        raise ImportValidationError(
            f"{name}: expected a .csv file, got '{file.filename}'"
        )

    # dtype=str + keep_default_na=False: read every cell as the exact text in the file, so empty
    # cells stay "" (not NaN) and the type checks below see what the user actually wrote.
    # utf-8-sig strips the byte-order mark Excel adds, which would otherwise corrupt the first
    # column name.
    try:
        df = pd.read_csv(
            file.stream, dtype=str, keep_default_na=False, encoding="utf-8-sig"
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        raise ImportValidationError(f"{name}.csv is not a readable CSV file: {e}")

    df.columns = df.columns.str.strip()
    # Short rows come back as NaN even with keep_default_na=False
    df = df.fillna("")
    missing_columns = [col for col in schema if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in schema]
    if missing_columns or extra_columns:
        raise ImportValidationError(
            f"{name}.csv has wrong columns. Missing: {missing_columns}. Unexpected: {extra_columns}. "
            f"Expected: {list(schema)}"
        )
    if df.empty:
        raise ImportValidationError(f"{name}.csv contains no data rows")

    # Error messages give the line number in the file: index + 2, because the index starts at 0
    # and line 1 is the header. Only the first problem is reported, which keeps messages short.
    for col, (type, required) in schema.items():
        s = df[col].str.strip()
        blank = s == ""
        if required and blank.any():
            raise ImportValidationError(
                f"{name}.csv: '{col}' is empty at line {blank.idxmax() + 2}"
            )
        if type == "int":
            bad = ~blank & ~s.str.fullmatch(INT_REGEX)
        elif type == "float":
            bad = ~blank & ~s.str.fullmatch(FLOAT_REGEX)
        else:
            bad = pd.Series(False, index=s.index)
        if bad.any():
            raise ImportValidationError(
                f"{name}.csv: '{col}' must be {type}, found '{s[bad].iloc[0]}' "
                f"at line {bad.idxmax() + 2}"
            )

        df[col] = (
            s.astype(int)
            if type == "int"
            else s.astype(float)
            if type == "float"
            else s
        )

        if col in RANGES:
            lo, hi = RANGES[col]
            out = ~df[col].between(lo, hi)
            if out.any():
                raise ImportValidationError(
                    f"{name}.csv: '{col}' must be between {lo} and {hi}, "
                    f"found {df[col][out].iloc[0]} at line {out.idxmax() + 2}"
                )

    for col in UNIQUE[name]:
        dup = df[col].duplicated()
        if dup.any():
            raise ImportValidationError(
                f"{name}.csv: duplicate {col} '{df[col][dup].iloc[0]}' "
                f"at line {dup.idxmax() + 2}"
            )

    return df


def lookup(frames, dataset, id_col, iata_col):
    """Return as {id: iata} for a dataset: from the upload if it is in this request, otherwise from the DB."""
    if dataset in frames:
        return dict(zip(frames[dataset][id_col], frames[dataset][iata_col]))

    model = MODELS[dataset]
    rows = db.session.execute(
        db.select(getattr(model, id_col), getattr(model, iata_col))
    ).all()

    return {r[0]: r[1] for r in rows}


def routes_view(frames):
    """The routes as they will be after the import: the uploaded ones, othewise the stored ones."""
    if "routes" in frames:
        return frames["routes"]
    cols = ["route_id"] + [c for link in ROUTE_LINKS for c in link[:2]]
    rows = db.session.execute(db.select(*[getattr(Route, c) for c in cols])).all()
    return pd.DataFrame([tuple(r) for r in rows], columns=cols)


def check_integrity(frames):
    """Check the combined result of the request (not only just the DB, or just the files)."""
    routes = routes_view(frames)
    if routes.empty:
        return

    for id_col, iata_col, dataset, id_ref, iata_ref in ROUTE_LINKS:
        ref = lookup(frames, dataset, id_ref, iata_ref)

        orphan = ~routes[id_col].isin(list(ref))
        if orphan.any():
            bad = routes[orphan].iloc[0]
            raise ImportValidationError(
                f"{int(orphan.sum())} route(s) would reference a non-existent "
                f"{dataset[:-1]} (e.g. route_id {bad['route_id']} has {id_col}={bad[id_col]})",
                409,
            )

        # 400 when the uploaded routes file contradicts itself/its references (a bad file);
        # 409 when stored routes would clash with a newly uploaded airports/airlines file
        # (a conflict with existing data rather than a malformed request).
        mismatch = routes[id_col].map(ref) != routes[iata_col]
        if mismatch.any():
            bad = routes[mismatch].iloc[0]
            raise ImportValidationError(
                f"{int(mismatch.sum())} route(s) have inconsistent {id_col}/{iata_col} "
                f"(e.g. route_id {bad['route_id']}: {id_col}={bad[id_col]} "
                f"is not '{bad[iata_col]}')",
                400 if "routes" in frames else 409,
            )


def replace_data(frames):
    """Replace the uploaded datasets inside ONE transaction (adhere to atomicity principle as required by the spec)."""
    # Everything runs in the session's single transaction and is committed once at the end; any
    # error rolls back, so the database is either fully updated or untouched.
    # Deletes go child-first (routes before the datasets they reference) and inserts go
    # parent-first, so no intermediate state has routes pointing at missing rows.
    try:
        for name in ("routes", "airports", "airlines"):
            if name in frames:
                db.session.execute(db.delete(MODELS[name]))

        for name in ("airlines", "airports", "routes"):
            if name in frames:
                # Empty optional cells (e.g. codeshare) are stored as NULL, not ""
                records = [
                    {k: (None if v == "" else v) for k, v in row.items()}
                    for row in frames[name].to_dict("records")
                ]
                db.session.execute(db.insert(MODELS[name]), records)

        db.session.commit()
    except Exception:
        db.session.rollback()
        raise


def check_travel_plans(frames):
    """Reject (409) a routes import that would remove any route_id still used by a travel plan.

    Routes are matched by route_id only. The error names the affected accounts and how many
    of their plans are affected (never plan contents), so the admin can ask them to clean up.
    """
    if "routes" not in frames:
        return

    new_route_ids = set(frames["routes"]["route_id"])
    rows = db.session.execute(
        db.select(TravelPlanRoute.route_id, TravelPlan.id, User.username)
        .join(TravelPlan, TravelPlanRoute.plan_id == TravelPlan.id)
        .join(User, TravelPlan.owner_id == User.id)
    ).all()
    affected = [row for row in rows if row.route_id not in new_route_ids]
    if not affected:
        return

    missing_route_ids = sorted({row.route_id for row in affected})
    plans_per_user = Counter(
        username for username, _ in {(row.username, row.id) for row in affected}
    )
    raise ImportValidationError(
        f"Import would remove {len(missing_route_ids)} route(s) still used by "
        f"{sum(plans_per_user.values())} travel plan(s). The affected accounts must remove "
        "these routes from their plans before routes.csv can be re-uploaded.",
        409,
        missing_route_ids=missing_route_ids,
        affected_accounts=[
            {"username": name, "plans": count}
            for name, count in sorted(plans_per_user.items())
        ],
    )


@dataset_ns.route("")
class FileUpload(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @dataset_ns.expect(upload_parser)
    @dataset_ns.doc(
        description=(
            "Import any subset of `airports.csv`, `airlines.csv`, `routes.csv` "
            "(multipart/form-data; any field name as long as the file is called exactly airports.csv / airlines.csv / routes.csv).\n\n"
            "Each uploaded file **replaces all existing records** of its dataset; datasets not "
            "in the request are untouched. Every file is validated (columns, types, unique ids, "
            "id/iata consistency) before anything is written, referential integrity is checked "
            "on the combined result of the request, and the import is atomic.\n\n"
            "**Existing travel plans:** an import that includes `routes.csv` is **rejected with 409** "
            "if any travel plan (of any account) references a `route_id` that is not in the new file. "
            "Nothing is changed, and the response lists `missing_route_ids` and `affected_accounts` "
            "(username + number of affected plans, never plan contents, so plans stay private). "
            "The admin cannot edit other users' plans, so the owners must remove those routes first; "
            "the admin then uploads the file again (rejected uploads are not kept). Routes are matched by "
            "`route_id` only, so a route that keeps its id but changes details stays in the plans. "
            "Imports of only `airports.csv` / `airlines.csv` cannot remove routes, so they are never "
            "blocked by travel plans.\n\n"
            "**Access:** Admin only"
        )
    )
    @dataset_ns.marshal_with(import_response_model, code=200, skip_none=True)
    @dataset_ns.response(200, "Import successful")
    @dataset_ns.response(
        400, "No files / malformed file / schema or consistency violation"
    )
    @dataset_ns.response(401, "Unauthorised access, please log in")
    @dataset_ns.response(403, "Insufficient permission and/or deactivated account")
    @dataset_ns.response(409, "Referential integrity (or travel plan) conflict")
    def post(self):
        """Import airports, airlines and/or routes from CSV files"""
        # Phase 1 validates everything (each file, cross-file integrity, travel plans) without
        # touching the database; phase 2 writes only if all of it passed. dataset_ns.abort()
        # raises, so `frames` is always defined when phase 2 runs.
        try:
            uploads = {}

            for _field, file in request.files.items(multi=True):
                if not file.filename:
                    continue

                name = dataset_name_matcher(file)
                if name in uploads:
                    raise ImportValidationError(f"{name} was uploaded more than once")

                uploads[name] = file

            if not uploads:
                raise ImportValidationError("No files uploaded")

            frames = {
                name: parse_and_validate_uploaded_file(name, file)
                for name, file in uploads.items()
            }

            check_integrity(frames)
            check_travel_plans(frames)
        except ImportValidationError as e:
            dataset_ns.abort(e.status, e.message, **e.details)

        try:
            replace_data(frames)
        except SQLAlchemyError:
            logger.exception("CSV import failed, transaction rolled back")
            dataset_ns.abort(500, "Import failed, no changes were made")

        logger.info("Imported datasets: {}", {n: len(df) for n, df in frames.items()})
        return {
            "message": "Import successful",
            "imported": {name: len(df) for name, df in frames.items()},
        }, 200


# --------------------------------------------------------------------------------
# Requirement Set 3: Access data/info about imported Airports, Airlines and Routes
# --------------------------------------------------------------------------------
READ_ACCESS = (
    "\n\n**Access:** any active account (Admin or User). "
    "Data is empty until the Admin imports it via `POST /imports`."
)
MATCHING_NOTE = (
    "\n\n**Matching:** text filters (name, city, country) are case-insensitive partial matches "
    "(`aus` matches *Australia*); code filters are case-insensitive exact matches. Filters "
    "combine with AND. Fuzzy matching is not used: codes are identifiers where a near miss "
    "is a different airport/airline, and partial matching already covers incomplete names."
)

# ---- Response models ----
# Small summaries embedded in routes so a client sees names without extra calls,
# plus an href to follow for the full record.
# Links are computed per object with `attribute=lambda obj: ...` and api.url_for(): the URL is
# built from the Resource class at request time, so it always matches the real route (prefix
# included) and the classes can be defined further down the file. Clients navigate by these
# links instead of building URLs from database ids themselves.
airport_summary_model = airport_ns.model(
    "AirportSummary",
    {
        "airport_id": fields.Integer(description="Airport id"),
        "iata_code": fields.String(description="3-letter IATA code, e.g. SYD"),
        "name": fields.String(description="Airport name"),
        "city": fields.String(description="City served"),
        "country": fields.String(description="Country"),
        "href": fields.String(
            attribute=lambda a: api.url_for(AirportItem, airport_id=a.airport_id),
            description="URL of the full airport record",
        ),
    },
)

airline_summary_model = airline_ns.model(
    "AirlineSummary",
    {
        "airline_id": fields.Integer(description="Airline id"),
        "iata_code": fields.String(description="2-letter IATA code, e.g. QF"),
        "name": fields.String(description="Airline name"),
        "href": fields.String(
            attribute=lambda a: api.url_for(AirlineItem, airline_id=a.airline_id),
            description="URL of the full airline record",
        ),
    },
)

airport_links_model = airport_ns.model(
    "AirportLinks",
    {
        "self": fields.String(
            attribute=lambda a: api.url_for(AirportItem, airport_id=a.airport_id)
        ),
        "airlines": fields.String(
            attribute=lambda a: api.url_for(AirportAirlines, airport_id=a.airport_id),
            description="Airlines operating from this airport",
        ),
        "departing_routes": fields.String(
            attribute=lambda a: api.url_for(RouteList, origin=a.iata_code),
            description="Routes departing from this airport",
        ),
        "arriving_routes": fields.String(
            attribute=lambda a: api.url_for(RouteList, destination=a.iata_code),
            description="Routes arriving at this airport",
        ),
    },
)

airport_model = airport_ns.model(
    "Airport",
    {
        "airport_id": fields.Integer(description="Airport id"),
        "name": fields.String(description="Airport name"),
        "city": fields.String(description="City served"),
        "country": fields.String(description="Country"),
        "iata_code": fields.String(description="3-letter IATA code"),
        "icao_code": fields.String(description="4-letter ICAO code"),
        "latitude": fields.Float(description="Latitude in degrees"),
        "longitude": fields.Float(description="Longitude in degrees"),
        "altitude": fields.Integer(description="Altitude in feet"),
        "timezone": fields.String(description="IANA timezone, e.g. Australia/Sydney"),
        # attribute=lambda a: a passes the airport itself to the nested links model
        "_links": fields.Nested(
            airport_links_model,
            attribute=lambda a: a,
            description="Related resources",
        ),
    },
)

airline_links_model = airline_ns.model(
    "AirlineLinks",
    {
        "self": fields.String(
            attribute=lambda a: api.url_for(AirlineItem, airline_id=a.airline_id)
        ),
        "routes": fields.String(
            attribute=lambda a: api.url_for(RouteList, airline=a.iata_code),
            description="Routes operated by this airline",
        ),
    },
)

airline_model = airline_ns.model(
    "Airline",
    {
        "airline_id": fields.Integer(description="Airline id"),
        "name": fields.String(description="Airline name"),
        "iata_code": fields.String(description="2-letter IATA code"),
        "icao_code": fields.String(description="3-letter ICAO code"),
        "country": fields.String(description="Country of the airline"),
        "_links": fields.Nested(
            airline_links_model,
            attribute=lambda a: a,
            description="Related resources",
        ),
    },
)

route_links_model = route_ns.model(
    "RouteLinks",
    {
        "self": fields.String(
            attribute=lambda r: api.url_for(RouteItem, route_id=r.route_id)
        ),
    },
)

route_model = route_ns.model(
    "Route",
    {
        "route_id": fields.Integer(description="Route id"),
        "airline": fields.Nested(
            airline_summary_model, description="Airline operating the route"
        ),
        "origin": fields.Nested(airport_summary_model, description="Departure airport"),
        "destination": fields.Nested(
            airport_summary_model, description="Arrival airport"
        ),
        "codeshare": fields.Boolean(
            attribute=lambda r: r.codeshare == "Y",
            description="True if the flight is a codeshare operated by another carrier",
        ),
        "stops": fields.Integer(description="Number of stops (0 = direct)"),
        "equipment": fields.String(description="Aircraft type code, e.g. A320, 789"),
        "_links": fields.Nested(
            route_links_model, attribute=lambda r: r, description="Related resources"
        ),
    },
)

airport_page_model = page_model(airport_ns, "AirportPage", "airports", airport_model)
airline_page_model = page_model(airline_ns, "AirlinePage", "airlines", airline_model)
route_page_model = page_model(route_ns, "RoutePage", "routes", route_model)

# ---- Query parameters ----
airport_list_parser = pagination_parser.copy()
airport_list_parser.add_argument(
    "name", location="args", help="Partial airport name, e.g. kingsford"
)
airport_list_parser.add_argument("city", location="args", help="Partial city, e.g. syd")
airport_list_parser.add_argument(
    "country", location="args", help="Partial country, e.g. aus"
)
airport_list_parser.add_argument(
    "code", location="args", help="Exact IATA or ICAO code, e.g. SYD or YSSY"
)

airline_list_parser = pagination_parser.copy()
airline_list_parser.add_argument(
    "name", location="args", help="Partial airline name, e.g. qan"
)
airline_list_parser.add_argument(
    "country", location="args", help="Partial country, e.g. aus"
)
airline_list_parser.add_argument(
    "code", location="args", help="Exact IATA or ICAO code, e.g. QF or QFA"
)

route_list_parser = pagination_parser.copy()
route_list_parser.add_argument(
    "airline", location="args", help="Exact airline IATA code, e.g. QF"
)
route_list_parser.add_argument(
    "origin", location="args", help="Exact origin airport IATA code, e.g. SYD"
)
route_list_parser.add_argument(
    "destination", location="args", help="Exact destination airport IATA code, e.g. MEL"
)
route_list_parser.add_argument(
    "origin_country", location="args", help="Partial origin country, e.g. aus"
)
route_list_parser.add_argument(
    "destination_country", location="args", help="Partial destination country, e.g. jap"
)
route_list_parser.add_argument(
    "direct",
    type=inputs.boolean,
    location="args",
    help="true = non-stop routes only, false = routes with stops only",
)
route_list_parser.add_argument(
    "codeshare", type=inputs.boolean, location="args", help="true / false"
)
route_list_parser.add_argument(
    "equipment", location="args", help="Exact aircraft type code, e.g. A320"
)


# ---- Airport-related information ----
@airport_ns.route("")
class AirportList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @airport_ns.expect(airport_list_parser)
    @airport_ns.doc(
        description="Retrieve a page of airports ordered by id, optionally filtered."
        + MATCHING_NOTE
        + READ_ACCESS
    )
    @airport_ns.marshal_with(airport_page_model, code=200)
    @airport_ns.response(200, "A page of airports")
    @airport_ns.response(400, "Invalid query parameter")
    @airport_ns.response(401, "Unauthorised access, please log in")
    @airport_ns.response(403, "Deactivated account")
    def get(self):
        """Retrieve a paginated list of airports"""
        args = airport_list_parser.parse_args()
        statement = db.select(Airport).order_by(Airport.airport_id)

        for col in ("name", "city", "country"):
            if args[col]:
                statement = statement.where(contains(getattr(Airport, col), args[col]))
        if args["code"]:
            code = args["code"].strip().upper()
            statement = statement.where(
                db.or_(Airport.iata_code == code, Airport.icao_code == code)
            )

        return paginate(statement, args, "airports"), 200


@airport_ns.route("/<int:airport_id>")
@airport_ns.param("airport_id", "The airport's unique ID")
class AirportItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @airport_ns.doc(
        description="Retrieve one airport. `_links` points to the airlines operating from it "
        "and to its departing / arriving routes." + READ_ACCESS
    )
    @airport_ns.marshal_with(airport_model, code=200)
    @airport_ns.response(200, "Airport details")
    @airport_ns.response(401, "Unauthorised access, please log in")
    @airport_ns.response(403, "Deactivated account")
    @airport_ns.response(404, "Airport not found")
    def get(self, airport_id):
        """Retrieve a specific airport"""
        airport = db.session.get(Airport, airport_id)
        if airport is None:
            airport_ns.abort(404, f"Airport {airport_id} not found")

        return airport, 200


@airport_ns.route("/<int:airport_id>/airlines")
@airport_ns.param("airport_id", "The airport's unique ID")
class AirportAirlines(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @airport_ns.expect(pagination_parser)
    @airport_ns.doc(
        description="Retrieve the airlines operating from this airport, i.e. airlines with at "
        "least one route **departing** from it, ordered by id." + READ_ACCESS
    )
    @airport_ns.marshal_with(airline_page_model, code=200)
    @airport_ns.response(200, "A page of airlines")
    @airport_ns.response(400, "Invalid query parameter")
    @airport_ns.response(401, "Unauthorised access, please log in")
    @airport_ns.response(403, "Deactivated account")
    @airport_ns.response(404, "Airport not found")
    def get(self, airport_id):
        """Retrieve the airlines operating from an airport"""
        args = pagination_parser.parse_args()
        if db.session.get(Airport, airport_id) is None:
            airport_ns.abort(404, f"Airport {airport_id} not found")

        # "Operating from" = has at least one route departing this airport. A sub-query with IN
        # returns each airline once, however many routes it flies from here.
        departing_airlines = db.select(Route.airline_id).where(
            Route.origin_airport_id == airport_id
        )
        statement = (
            db.select(Airline)
            .where(Airline.airline_id.in_(departing_airlines))
            .order_by(Airline.airline_id)
        )
        return paginate(statement, args, "airlines"), 200


# ---- Airline-related information ----
@airline_ns.route("")
class AirlineList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @airline_ns.expect(airline_list_parser)
    @airline_ns.doc(
        description="Retrieve a page of airlines ordered by id, optionally filtered."
        + MATCHING_NOTE
        + READ_ACCESS
    )
    @airline_ns.marshal_with(airline_page_model, code=200)
    @airline_ns.response(200, "A page of airlines")
    @airline_ns.response(400, "Invalid query parameter")
    @airline_ns.response(401, "Unauthorised access, please log in")
    @airline_ns.response(403, "Deactivated account")
    def get(self):
        """Retrieve a paginated list of airlines"""
        args = airline_list_parser.parse_args()
        statement = db.select(Airline).order_by(Airline.airline_id)

        for col in ("name", "country"):
            if args[col]:
                statement = statement.where(contains(getattr(Airline, col), args[col]))
        if args["code"]:
            code = args["code"].strip().upper()
            statement = statement.where(
                db.or_(Airline.iata_code == code, Airline.icao_code == code)
            )

        return paginate(statement, args, "airlines"), 200


@airline_ns.route("/<int:airline_id>")
@airline_ns.param("airline_id", "The airline's unique ID")
class AirlineItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @airline_ns.doc(
        description="Retrieve one airline. `_links.routes` lists the routes it operates."
        + READ_ACCESS
    )
    @airline_ns.marshal_with(airline_model, code=200)
    @airline_ns.response(200, "Airline details")
    @airline_ns.response(401, "Unauthorised access, please log in")
    @airline_ns.response(403, "Deactivated account")
    @airline_ns.response(404, "Airline not found")
    def get(self, airline_id):
        """Retrieve a specific airline"""
        airline = db.session.get(Airline, airline_id)
        if airline is None:
            airline_ns.abort(404, f"Airline {airline_id} not found")

        return airline, 200


# ---- Route-related information ----
@route_ns.route("")
class RouteList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @route_ns.expect(route_list_parser)
    @route_ns.doc(
        description="Retrieve a page of routes ordered by id, optionally filtered. "
        "Each route embeds a summary of its airline and airports with an `href` to the full record.\n\n"
        "Common lookups:\n"
        "- routes of an airline: `?airline=QF`\n"
        "- routes departing an airport: `?origin=SYD`\n"
        "- direct flights between two airports: `?origin=SYD&destination=MEL&direct=true`\n"
        "- international routes from a country: `?origin_country=aus&destination_country=jap`\n\n"
        "These are filters on one collection rather than separate sub-endpoints, so they can be "
        "combined freely; airport and airline records link here via `_links`."
        + MATCHING_NOTE
        + READ_ACCESS
    )
    @route_ns.marshal_with(route_page_model, code=200)
    @route_ns.response(200, "A page of routes")
    @route_ns.response(400, "Invalid query parameter")
    @route_ns.response(401, "Unauthorised access, please log in")
    @route_ns.response(403, "Deactivated account")
    def get(self):
        """Retrieve a paginated list of routes"""
        args = route_list_parser.parse_args()
        statement = db.select(Route).order_by(Route.route_id)

        # Code filters compare against the route's own IATA columns (guaranteed consistent with
        # the airline/airport tables by the import), which avoids joins for the common lookups.
        for arg, col in (
            ("airline", Route.airline_iata),
            ("origin", Route.origin_iata),
            ("destination", Route.destination_iata),
        ):
            if args[arg]:
                statement = statement.where(col == args[arg].strip().upper())
        # .has() turns into an EXISTS sub-query on the related airport. Origin and destination
        # both point at the airports table, so a plain join would need table aliases.
        if args["origin_country"]:
            statement = statement.where(
                Route.origin.has(contains(Airport.country, args["origin_country"]))
            )
        if args["destination_country"]:
            statement = statement.where(
                Route.destination.has(
                    contains(Airport.country, args["destination_country"])
                )
            )
        # `is not None` because false is a meaningful filter value ("routes with stops only")
        if args["direct"] is not None:
            statement = statement.where(
                Route.stops == 0 if args["direct"] else Route.stops > 0
            )
        if args["codeshare"] is not None:
            statement = statement.where(
                Route.codeshare == "Y"
                if args["codeshare"]
                else Route.codeshare.is_(None)
            )
        if args["equipment"]:
            statement = statement.where(
                db.func.upper(Route.equipment) == args["equipment"].strip().upper()
            )

        return paginate(statement, args, "routes"), 200


@route_ns.route("/<int:route_id>")
@route_ns.param("route_id", "The route's unique ID")
class RouteItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @route_ns.doc(
        description="Retrieve one route with summaries of its airline, origin and destination; "
        "follow each `href` for the full record." + READ_ACCESS
    )
    @route_ns.marshal_with(route_model, code=200)
    @route_ns.response(200, "Route details")
    @route_ns.response(401, "Unauthorised access, please log in")
    @route_ns.response(403, "Deactivated account")
    @route_ns.response(404, "Route not found")
    def get(self, route_id):
        """Retrieve a specific route"""
        route = db.session.get(Route, route_id)
        if route is None:
            route_ns.abort(404, f"Route {route_id} not found")

        return route, 200


# ----------------------------------------------
# Requirement Set 4: My Travel Plans (private)
# ----------------------------------------------
PLAN_ACCESS = (
    "\n\n**Access:** any active account (Admin or User). Plans are private: you only ever see "
    "your own, and the admin has no access to other accounts' plans. A plan that does not "
    "belong to you returns **404**, exactly like a plan that does not exist, so plan ids of "
    "other accounts cannot be discovered."
)


def get_own_plan_or_404(plan_id):
    """Return the caller's plan, or 404 if it doesn't exist or belongs to someone else.

    Ownership is part of the query itself rather than checked afterwards, so there is no code
    path that loads another account's plan. This applies to the admin too (spec RS1/RS4).
    """
    plan = db.session.scalars(
        db.select(TravelPlan).where(
            TravelPlan.id == plan_id, TravelPlan.owner_id == g.current_user.id
        )
    ).first()
    if plan is None:
        plan_ns.abort(404, f"Travel plan {plan_id} not found")
    return plan


def parse_date(value, field):
    """Parse a YYYY-MM-DD string (Swagger's date format is not enforced by validation)."""
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        plan_ns.abort(
            400, f"'{field}' must be a date in YYYY-MM-DD format, got '{value}'"
        )


def check_dates(departure_date, return_date):
    if return_date < departure_date:
        plan_ns.abort(400, "'return_date' must not be earlier than 'departure_date'")


def check_route_ids_exist(route_ids):
    found = set(
        db.session.scalars(
            db.select(Route.route_id).where(Route.route_id.in_(route_ids))
        )
    )
    unknown = sorted(set(route_ids) - found)
    if unknown:
        plan_ns.abort(400, f"Unknown route id(s): {unknown}")


def clean_name(value):
    name = value.strip()
    if not name:
        plan_ns.abort(400, "'name' must not be blank")
    return name


# ---- Request / response models ----
plan_create_model = plan_ns.model(
    "TravelPlanCreate",
    {
        "name": fields.String(
            required=True, min_length=1, max_length=100, description="Plan name"
        ),
        "departure_date": fields.Date(
            required=True, description="Travel departure date (YYYY-MM-DD)"
        ),
        "return_date": fields.Date(
            required=True,
            description="Travel return date (YYYY-MM-DD), not earlier than departure_date",
        ),
        "route_ids": fields.List(
            fields.Integer,
            description="Optional initial routes (ids from GET /routes); duplicates are ignored",
        ),
    },
    strict=True,
)

plan_update_model = plan_ns.model(
    "TravelPlanUpdate",
    {
        "name": fields.String(
            min_length=1, max_length=100, description="New plan name"
        ),
        "departure_date": fields.Date(description="New departure date (YYYY-MM-DD)"),
        "return_date": fields.Date(description="New return date (YYYY-MM-DD)"),
    },
    strict=True,
)

plan_links_model = plan_ns.model(
    "TravelPlanLinks",
    {
        "self": fields.String(
            attribute=lambda p: api.url_for(TravelPlanItem, plan_id=p.id)
        ),
        "routes": fields.String(
            attribute=lambda p: api.url_for(TravelPlanRouteList, plan_id=p.id),
            description="Full details of the plan's routes, with their airlines and airports",
        ),
    },
)

plan_model = plan_ns.model(
    "TravelPlan",
    {
        "id": fields.Integer(description="Travel plan id"),
        "name": fields.String(description="Plan name"),
        "departure_date": fields.Date(description="Travel departure date"),
        "return_date": fields.Date(description="Travel return date"),
        "route_ids": fields.List(
            fields.Integer, description="Ids of the routes in this plan"
        ),
        "_links": fields.Nested(
            plan_links_model, attribute=lambda p: p, description="Related resources"
        ),
    },
)

plan_page_model = page_model(plan_ns, "TravelPlanPage", "travel_plans", plan_model)
plan_list_parser = pagination_parser.copy()


@plan_ns.route("")
class TravelPlanList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @plan_ns.expect(plan_list_parser)
    @plan_ns.doc(
        description="Retrieve a page of your travel plans ordered by departure date. "
        "Each plan lists its `route_ids`; follow `_links.routes` for full route, airline and "
        "airport details." + PLAN_ACCESS
    )
    @plan_ns.marshal_with(plan_page_model, code=200)
    @plan_ns.response(200, "A page of your travel plans")
    @plan_ns.response(400, "Invalid page or per_page")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    def get(self):
        """Retrieve your travel plans"""
        args = plan_list_parser.parse_args()
        statement = (
            db.select(TravelPlan)
            .where(TravelPlan.owner_id == g.current_user.id)
            .order_by(TravelPlan.departure_date, TravelPlan.id)
        )
        return paginate(statement, args, "travel_plans"), 200

    @plan_ns.expect(plan_create_model, validate=True)
    @plan_ns.doc(
        description="Create a travel plan owned by you. Dates use `YYYY-MM-DD`; past dates are "
        "allowed, but `return_date` must not be earlier than `departure_date`. `route_ids` is "
        "optional; routes can also be added later with `PUT /travel-plans/{plan_id}/routes/{route_id}`. "
        "The response has a `Location` header pointing to the new plan." + PLAN_ACCESS
    )
    @plan_ns.marshal_with(plan_model, code=201)
    @plan_ns.response(201, "Travel plan created")
    @plan_ns.response(400, "Invalid payload, dates or unknown route ids")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    def post(self):
        """Create a travel plan"""
        data = plan_ns.payload
        departure_date = parse_date(data["departure_date"], "departure_date")
        return_date = parse_date(data["return_date"], "return_date")
        check_dates(departure_date, return_date)

        # A plan holds a *set* of routes, so duplicate ids in the request are simply merged
        route_ids = sorted(set(data.get("route_ids") or []))
        check_route_ids_exist(route_ids)

        plan = TravelPlan(
            owner_id=g.current_user.id,
            name=clean_name(data["name"]),
            departure_date=departure_date,
            return_date=return_date,
            route_links=[TravelPlanRoute(route_id=rid) for rid in route_ids],
        )
        db.session.add(plan)
        db.session.commit()

        # Location header: standard for 201, tells the client where the new resource lives
        return plan, 201, {"Location": api.url_for(TravelPlanItem, plan_id=plan.id)}


@plan_ns.route("/<int:plan_id>")
@plan_ns.param("plan_id", "The travel plan's unique ID")
class TravelPlanItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @plan_ns.doc(description="Retrieve one of your travel plans." + PLAN_ACCESS)
    @plan_ns.marshal_with(plan_model, code=200)
    @plan_ns.response(200, "Travel plan details")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    @plan_ns.response(404, "Travel plan not found")
    def get(self, plan_id):
        """Retrieve a specific travel plan"""
        return get_own_plan_or_404(plan_id), 200

    @plan_ns.expect(plan_update_model, validate=True)
    @plan_ns.doc(
        description="Update the name and/or dates of one of your travel plans. Send only the "
        "fields to change; the date rule is checked against the resulting dates. Routes are "
        "managed through `/travel-plans/{plan_id}/routes`." + PLAN_ACCESS
    )
    @plan_ns.marshal_with(plan_model, code=200)
    @plan_ns.response(200, "Travel plan updated")
    @plan_ns.response(400, "Empty payload or invalid dates")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    @plan_ns.response(404, "Travel plan not found")
    def patch(self, plan_id):
        """Update a travel plan"""
        plan = get_own_plan_or_404(plan_id)
        data = plan_ns.payload
        if not data:
            plan_ns.abort(
                400, "Provide at least one of: name, departure_date, return_date"
            )

        # Validate the dates the plan would end up with: changing only one date must still keep
        # return_date >= departure_date with the other, unchanged date.
        departure_date = plan.departure_date
        return_date = plan.return_date
        if "departure_date" in data:
            departure_date = parse_date(data["departure_date"], "departure_date")
        if "return_date" in data:
            return_date = parse_date(data["return_date"], "return_date")
        check_dates(departure_date, return_date)

        if "name" in data:
            plan.name = clean_name(data["name"])
        plan.departure_date = departure_date
        plan.return_date = return_date
        db.session.commit()

        return plan, 200

    @plan_ns.doc(
        description="Delete one of your travel plans (routes themselves are not affected)."
        + PLAN_ACCESS
    )
    @plan_ns.response(204, "Travel plan deleted")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    @plan_ns.response(404, "Travel plan not found")
    def delete(self, plan_id):
        """Delete a travel plan"""
        db.session.delete(get_own_plan_or_404(plan_id))
        db.session.commit()

        return "", 204


@plan_ns.route("/<int:plan_id>/routes")
@plan_ns.param("plan_id", "The travel plan's unique ID")
class TravelPlanRouteList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @plan_ns.expect(pagination_parser)
    @plan_ns.doc(
        description="Retrieve a page of the routes in one of your plans, ordered by route id. "
        "Each route embeds its airline, origin and destination with an `href` to the full "
        "record, in the same format as `GET /routes`." + PLAN_ACCESS
    )
    @plan_ns.marshal_with(route_page_model, code=200)
    @plan_ns.response(200, "A page of the plan's routes")
    @plan_ns.response(400, "Invalid page or per_page")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    @plan_ns.response(404, "Travel plan not found")
    def get(self, plan_id):
        """Retrieve the routes in a travel plan"""
        args = pagination_parser.parse_args()
        plan = get_own_plan_or_404(plan_id)
        statement = (
            db.select(Route)
            .join(TravelPlanRoute, TravelPlanRoute.route_id == Route.route_id)
            .where(TravelPlanRoute.plan_id == plan.id)
            .order_by(Route.route_id)
        )
        return paginate(statement, args, "routes"), 200


@plan_ns.route("/<int:plan_id>/routes/<int:route_id>")
@plan_ns.param("plan_id", "The travel plan's unique ID")
@plan_ns.param("route_id", "The route's unique ID (from GET /routes)")
class TravelPlanRouteItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @plan_ns.doc(
        description="Add a route to one of your plans. Idempotent: adding a route that is "
        "already in the plan changes nothing and still returns 200." + PLAN_ACCESS
    )
    @plan_ns.marshal_with(plan_model, code=200)
    @plan_ns.response(200, "Route is in the plan; returns the updated plan")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    @plan_ns.response(404, "Travel plan or route not found")
    # PUT on the member URL (rather than POST to the collection) because adding a route is
    # idempotent: the client names the exact route, and repeating the request changes nothing.
    def put(self, plan_id, route_id):
        """Add a route to a travel plan"""
        plan = get_own_plan_or_404(plan_id)
        if db.session.get(Route, route_id) is None:
            plan_ns.abort(404, f"Route {route_id} not found")

        if route_id not in plan.route_ids:
            plan.route_links.append(TravelPlanRoute(route_id=route_id))
            db.session.commit()

        return plan, 200

    @plan_ns.doc(
        description="Remove a route from one of your plans (the route itself is not deleted)."
        + PLAN_ACCESS
    )
    @plan_ns.response(204, "Route removed from the plan")
    @plan_ns.response(401, "Unauthorised access, please log in")
    @plan_ns.response(403, "Deactivated account")
    @plan_ns.response(404, "Travel plan not found, or route not in this plan")
    def delete(self, plan_id, route_id):
        """Remove a route from a travel plan"""
        plan = get_own_plan_or_404(plan_id)
        link = next((l for l in plan.route_links if l.route_id == route_id), None)
        if link is None:
            plan_ns.abort(404, f"Route {route_id} is not in travel plan {plan_id}")

        plan.route_links.remove(link)
        db.session.commit()

        return "", 204


# ---------------------------------------------
# Requirement Set 5: Exporting travel plans
# ---------------------------------------------
EXPORT_FORMATS = ("json", "png")
# The visualisation shows at most this many upcoming plans (as required by the spec)
MAX_PLANS_IN_IMAGE = 5
# Routes listed under each plan in the image before collapsing into "+N more"
MAX_ROUTES_IN_LABEL = 3

# The file must be self-contained, so airlines and airports are exported in full and
# without `_links`/`href` (relative API URLs mean nothing once the file is downloaded)
export_airline_model = export_ns.model(
    "ExportAirline", {k: v for k, v in airline_model.items() if k != "_links"}
)
export_airport_model = export_ns.model(
    "ExportAirport", {k: v for k, v in airport_model.items() if k != "_links"}
)

export_route_model = export_ns.model(
    "ExportRoute",
    {
        "route_id": fields.Integer(description="Route id"),
        "airline": fields.Nested(
            export_airline_model, description="Full airline record"
        ),
        "origin": fields.Nested(
            export_airport_model, description="Full departure airport record"
        ),
        "destination": fields.Nested(
            export_airport_model, description="Full arrival airport record"
        ),
        "codeshare": route_model["codeshare"],
        "stops": route_model["stops"],
        "equipment": route_model["equipment"],
    },
)

export_plan_model = export_ns.model(
    "ExportTravelPlan",
    {
        "id": fields.Integer(description="Travel plan id"),
        "name": fields.String(description="Plan name"),
        "departure_date": fields.Date(description="Travel departure date"),
        "return_date": fields.Date(description="Travel return date"),
        "routes": fields.List(
            fields.Nested(export_route_model),
            description="Full details of every route in the plan, ordered by route id",
        ),
    },
)

export_file_model = export_ns.model(
    "TravelPlanExport",
    {
        "owner": fields.String(
            description="Username of the account that owns the plans"
        ),
        "exported_at": fields.DateTime(
            dt_format="iso8601", description="When the file was generated (UTC)"
        ),
        "total_plans": fields.Integer(description="Number of plans in the file"),
        "travel_plans": fields.List(
            fields.Nested(export_plan_model),
            description="All of your plans, ordered by departure date (then id)",
        ),
    },
)

export_parser = reqparse.RequestParser()
export_parser.add_argument(
    "format",
    required=True,
    choices=EXPORT_FORMATS,
    location="args",
    help="Output format (required): `json` = file with all plans, `png` = timeline image "
    "of the next 5 future plans",
)


def plan_label(plan, today):
    """Three-line y-axis label: name, dates + countdown, and a short route summary."""
    days_away = (plan.departure_date - today).days
    routes = [
        f"{r.origin.iata_code}\u2192{r.destination.iata_code} ({r.airline.iata_code})"
        for r in plan.routes
    ]
    if not routes:
        route_text = "No routes added yet"
    elif len(routes) > MAX_ROUTES_IN_LABEL:
        route_text = ", ".join(routes[:MAX_ROUTES_IN_LABEL])
        route_text += f" +{len(routes) - MAX_ROUTES_IN_LABEL} more"
    else:
        route_text = ", ".join(routes)

    return (
        f"{plan.name}\n"
        f"{plan.departure_date:%d %b %Y} \u2192 {plan.return_date:%d %b %Y} "
        f"(in {days_away} day{'s' if days_away != 1 else ''})\n"
        f"{route_text}"
    )


def render_plan_timeline(plans, username, today):
    """Draw upcoming plans as a Gantt-style timeline and return the PNG bytes.

    Uses matplotlib's object-oriented Figure API rather than pyplot: pyplot keeps global
    state and needs a GUI backend, which is unsafe inside a multi-threaded web server.
    """
    fig = Figure(figsize=(11, 1.4 * len(plans) + 2), dpi=110)
    ax = fig.add_subplot()
    colours = ["#2a6fdb", "#e07b39", "#3a9d5d", "#b8457a", "#7a5bc4"]

    for row, plan in enumerate(plans):
        # +1 so the bar covers the return day too (a same-day trip is still visible)
        days = (plan.return_date - plan.departure_date).days + 1
        ax.barh(
            row,
            days,
            left=mdates.date2num(plan.departure_date),
            height=0.5,
            color=colours[row % len(colours)],
        )
        ax.annotate(
            f"{days} day{'s' if days != 1 else ''}",
            xy=(mdates.date2num(plan.return_date) + 1, row),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    # Mark today so the user sees how far away each trip is
    ax.axvline(mdates.date2num(today), color="grey", linestyle="--", linewidth=1)
    ax.annotate(
        "Today",
        xy=(mdates.date2num(today), -0.6),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        fontsize=9,
        color="grey",
    )

    ax.set_yticks(range(len(plans)), [plan_label(p, today) for p in plans], fontsize=9)
    ax.set_ylim(len(plans) - 0.4, -0.9)  # nearest plan at the top
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    # Leave room on the right for the "N days" labels
    first = mdates.date2num(today) - 3
    last = mdates.date2num(max(p.return_date for p in plans)) + 1
    ax.set_xlim(first, last + (last - first) * 0.12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title(
        f"Next {len(plans)} upcoming travel plan{'s' if len(plans) != 1 else ''} "
        f"for {username}",
        fontsize=13,
        loc="left",
    )
    ax.set_xlabel(
        f"Bars run from departure to return date (inclusive). Generated on {today:%d %b %Y}."
    )

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    return buffer.getvalue()


@export_ns.route("/travel-plans")
class TravelPlanExport(Resource):
    method_decorators = [require_roles(UserRole.ADMIN, UserRole.USER)]

    @export_ns.expect(export_parser)
    @export_ns.produces(["application/json", "image/png"])
    @export_ns.doc(
        description=(
            "Export **your own travel plans**, either as a downloadable JSON file or as a "
            "timeline image.\n\n"
            "**Choosing the format:** the client must pass the `format` query parameter: "
            "`format=json` or `format=png`. A missing or unknown value returns 400 listing the "
            "supported formats; there is deliberately no default, so a client never receives a "
            "format it did not ask for. A query parameter is used rather than `Accept` header "
            "negotiation because it is explicit, visible in the URL, easy to try in Swagger UI "
            "and from a browser link, and new formats can be added later without changing "
            "existing clients' URLs.\n\n"
            "### `format=json`: all plans as a file\n"
            "All of your plans, ordered by departure date (ties by plan id). "
            "**File contents:** each plan includes the **full** details of its routes, and each "
            "route includes its full airline and origin/destination airport records, so the file "
            "can be used without calling the API again. API links (`href`, `_links`) are left "
            "out because they are meaningless outside the API. See the `TravelPlanExport` model "
            "below for the exact structure.\n\n"
            "**Download:** the response has `Content-Disposition: attachment` with a filename "
            "like `travel-plans_<username>_<YYYY-MM-DD>.json`, so browsers save it as a file "
            "(Swagger UI shows a *Download file* link). The whole export is returned in one file, "
            "so it is not paginated. `Cache-Control: no-store` keeps private plans out of caches.\n\n"
            "**No plans:** an account without plans still gets 200 and a valid file with "
            "`total_plans: 0` and an empty `travel_plans` list.\n\n"
            "### `format=png`: timeline of upcoming plans\n"
            "A PNG image (`image/png`, `Content-Disposition: inline`, so it is shown in the "
            "browser rather than downloaded; Swagger UI displays it directly) of up to **5 future "
            "plans with the nearest departure dates**. A plan is *future* when its departure date "
            "is strictly **after today** (server date), so a trip departing today is not included.\n\n"
            "The image is a Gantt-style timeline:\n"
            "- one horizontal bar per plan, from departure to return date (inclusive), nearest "
            "plan at the top, with the trip length (`N days`) next to the bar;\n"
            "- a dashed line marking today, so the gap to each trip is visible;\n"
            "- each plan's label shows its name, dates, how many days away it is, and its routes "
            "as `SYD\u2192HND (JL)` (first 3, then `+N more`).\n\n"
            "**No future plans:** returns **404** with a JSON message (not an empty image), "
            "because there is nothing to visualise; a client can check the status code before "
            "showing the image.\n\n"
            "**Access:** any active account (Admin or User). Only the caller's own plans are "
            "exported; the admin cannot export other accounts' plans."
        )
    )
    @export_ns.response(
        200,
        "`format=json`: JSON file with all of your plans. "
        "`format=png`: timeline image of your next 5 future plans",
        export_file_model,
    )
    @export_ns.response(400, "Missing or unsupported `format`")
    @export_ns.response(401, "Unauthorised access, please log in")
    @export_ns.response(403, "Deactivated account")
    @export_ns.response(404, "`format=png` only: you have no future travel plans")
    def get(self):
        """Export your travel plans as a JSON file or a timeline image"""
        args = export_parser.parse_args()
        user = g.current_user
        if args["format"] == "png":
            return self.png_timeline(user)

        # Reuses the RS3 field definitions through marshal(), so the export always has the same
        # field names and formats as GET /airports and GET /airlines.

        plans = db.session.scalars(
            db.select(TravelPlan)
            .where(TravelPlan.owner_id == user.id)
            .order_by(TravelPlan.departure_date, TravelPlan.id)
        ).all()

        exported_at = datetime.now(UTC)
        body = marshal(
            {
                "owner": user.username,
                "exported_at": exported_at,
                "total_plans": len(plans),
                "travel_plans": plans,
            },
            export_file_model,
        )

        safe_username = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in user.username
        )
        filename = f"travel-plans_{safe_username}_{exported_at:%Y-%m-%d}.json"
        # A raw Response (instead of returning a dict) so the Content-Disposition header makes
        # browsers save it as a file; indent=2 keeps the downloaded file human-readable.
        return Response(
            json.dumps(body, indent=2),
            status=200,
            mimetype="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    @staticmethod
    def png_timeline(user):
        # "Future" means departing strictly after today (server's local date), as in the spec.
        # The filtering, ordering and the limit of 5 are done in SQL.
        today = date.today()
        plans = db.session.scalars(
            db.select(TravelPlan)
            .where(TravelPlan.owner_id == user.id, TravelPlan.departure_date > today)
            .order_by(TravelPlan.departure_date, TravelPlan.id)
            .limit(MAX_PLANS_IN_IMAGE)
        ).all()
        if not plans:
            export_ns.abort(
                404,
                f"You have no future travel plans (departing after {today.isoformat()}) "
                "to visualise",
            )

        return Response(
            render_plan_timeline(plans, user.username, today),
            status=200,
            mimetype="image/png",
            headers={
                "Content-Disposition": f'inline; filename="travel-plans_{today:%Y-%m-%d}.png"',
                "Cache-Control": "no-store",
            },
        )


# ---------------------
# Run Flask application
# ---------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1", port=5000, debug=True
    )  # Be accessible via a browser at: http://127.0.0.1:5000
