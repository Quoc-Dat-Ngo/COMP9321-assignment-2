import os
from enum import StrEnum
from functools import wraps

import bcrypt
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt,
    get_jwt_identity,
    verify_jwt_in_request,
)
from flask_restx import Api, Resource, abort, fields, inputs, reqparse
from flask_sqlalchemy import SQLAlchemy
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage

load_dotenv()

app = Flask(__name__)
# Allow browser clients on other origins (e.g. upload_test.html opened from disk)
CORS(app)

# ---------------------------------
# Config SQLite database connection
# ---------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///app.db")
db = SQLAlchemy(app)

# ------------------------
# Configure JWT_SECRET_KEY
# ------------------------
# Utilise Python built-in 'secrets' library
app.config["JWT_SECRET_KEY"] = os.getenv(
    "JWT_SECRET_KEY", "282601b3f714615e2b6d371aa9f2db42240dd7a104c26b61ca3ddc497f04a61a"
)
# Let flask-jwt-extended errors (missing/expired token) reach its own handlers
# instead of being swallowed by flask-restx as a 500
app.config["PROPAGATE_EXCEPTIONS"] = True
# Keep 404 messages as written (no "did you mean ..." suffix from flask-restx)
app.config["ERROR_404_HELP"] = False
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


# Custom decorator to check the permission (i.e: Role) and the activation status of request user
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
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def contains(column, value):
    """Case-insensitive partial match; % and _ typed by the client are matched literally."""
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return column.ilike(f"%{escaped}%", escape="\\")


def paginate(statement, args, key):
    """Run a select for one page and wrap it in the shared page envelope."""
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


# Airline
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

    # Read-only links to the related rows (imports write the id columns directly)
    airline = db.relationship("Airline", lazy="joined", viewonly=True)
    origin = db.relationship(
        "Airport", foreign_keys=[origin_airport_id], lazy="joined", viewonly=True
    )
    destination = db.relationship(
        "Airport", foreign_keys=[destination_airport_id], lazy="joined", viewonly=True
    )

# Initiate SQLite database on first start-up,
# reuse existing database i.e: app.db for subsequent run
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
        if existing_user is None:
            auth_ns.abort(401, "Invalid username or password")

        if not existing_user.password_check(user_data.get("password")):
            auth_ns.abort(401, "Invalid username or password")

        if not existing_user.active:
            auth_ns.abort(403, "Deactivated account cannot access internal service")

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
        description="Delete a user with specified ID.\n\n**Access:** Admin only"
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

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


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
    try:
        for name in ("routes", "airports", "airlines"):
            if name in frames:
                db.session.execute(db.delete(MODELS[name]))

        for name in ("airlines", "airports", "routes"):
            if name in frames:
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
    """Requirement Set 4 hook. Policy: reject (409) a routes import that would remove
    any route_id still used by a travel plan; routes are matched by route_id only.

    TODO: once travel plans exist, compare the route ids used by plans with
    frames["routes"]["route_id"] and raise ImportValidationError(..., 409) listing the missing ids.
    """


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
            "if any travel plan (of any account) references a `route_id` that is not in the new file; "
            "the response lists the missing route ids and nothing is changed. Routes are matched by "
            "`route_id` only, so a route that keeps its id but changes details stays in the plans. "
            "Because plans are private, the admin cannot edit other users' plans: the owners must "
            "remove those routes first. Imports of only `airports.csv` / `airlines.csv` cannot remove "
            "routes, so they are never blocked by travel plans.\n\n"
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
            dataset_ns.abort(e.status, e.message)

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
# plus an href to follow for the full record
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

        for arg, col in (
            ("airline", Route.airline_iata),
            ("origin", Route.origin_iata),
            ("destination", Route.destination_iata),
        ):
            if args[arg]:
                statement = statement.where(col == args[arg].strip().upper())
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
        if args["direct"] is not None:
            statement = statement.where(
                Route.stops == 0 if args["direct"] else Route.stops > 0
            )
        if args["codeshare"] is not None:
            statement = statement.where(
                Route.codeshare == "Y" if args["codeshare"] else Route.codeshare.is_(None)
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


# ---------------------
# Run Flask application
# ---------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1", port=5000, debug=True
    )  # Be accessible via a browser at: http://127.0.0.1:5000
