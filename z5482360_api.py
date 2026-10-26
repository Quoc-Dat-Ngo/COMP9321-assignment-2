from enum import StrEnum
from functools import wraps

import bcrypt
from flask import Flask
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    verify_jwt_in_request,
)
from flask_restx import Api, Resource, abort, fields
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# ---------------------------------
# Config SQLite database connection
# ---------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
db = SQLAlchemy(app)

# ------------------------
# Configure JWT_SECRET_KEY
# ------------------------
# Utilise Python built-in 'secrets' library
app.config["JWT_SECRET_KEY"] = (
    "282601b3f714615e2b6d371aa9f2db42240dd7a104c26b61ca3ddc497f04a61a"
)
jwt = JWTManager(app)

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
users_ns = api.namespace("users", description="User management (for Admin only)")


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
            user = User.retrieve_user_by_username(get_jwt_identity())

            if user is None:
                abort(401, "Unauthorised access, please log in")
            if not user.active:
                abort(403, "This account has been deactivated")
            if claims.get("role") not in allowed_roles:
                abort(403, "Insufficient permission to perform this action")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# -----------------------------------------------------------
# Entity definition/schema for storing in the sqlite database
# -----------------------------------------------------------
# User
class User(db.Model):
    __tablename__ = "users"

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

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, unique=True, nullable=False)
    iata = db.Column(db.String(5), unique=True, nullable=False)
    icao = db.Column(db.String(5), unique=True, nullable=False)
    country = db.Column(db.Text, nullable=False)


# Airport
class Airport(db.Model):
    __tablename__ = "airports"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.Text, unique=True, nullable=False)
    city = db.Column(db.Text, nullable=False)
    country = db.Column(db.Text, nullable=False)
    iata = db.Column(db.String(5), unique=True, nullable=False)
    icao = db.Column(db.String(5), unique=True, nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lon = db.Column(db.Float, nullable=False)
    alt = db.Column(db.Integer, nullable=False)
    timezone = db.Column(db.Text, nullable=False)


# Route
class Route(db.Model):
    __tablename__ = "routes"

    id = db.Column(db.Integer, primary_key=True)
    airline_id = db.Column(db.Integer, db.ForeignKey("airlines.id"), nullable=False)
    airline_iata = db.Column(
        db.String(5), db.ForeignKey("airlines.iata"), nullable=False
    )
    src_airport_id = db.Column(db.Integer, db.ForeignKey("airports.id"), nullable=False)
    src_airport_iata = db.Column(
        db.String(5), db.ForeignKey("airports.iata"), nullable=False
    )
    des_airport_id = db.Column(db.Integer, db.ForeignKey("airports.id"), nullable=False)
    des_airport_iata = db.Column(
        db.String(5), db.ForeignKey("airports.iata"), nullable=False
    )
    codeshare = db.Column(db.String(5))
    stops = db.Column(db.Integer, nullable=False)
    equipment = db.Column(db.String(5), nullable=False)


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
        print("Automatically create 1 admin account and 1 user account.")

# ----------------------------------
# Model Schema for API documentation
# ----------------------------------
user_create_model = users_ns.model(
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

user_update_activation_model = users_ns.model(
    "UserActivationUpdate",
    {
        "active": fields.Boolean(
            description="Target status that admin wants to update", required=True
        )
    },
    strict=True,
)

user_response_model = users_ns.model(
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
        "refreshToken": fields.String(
            description="Refresh token to be used when access token expires"
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
        description="Entry point for a user to get into app, create access token and refresh token",
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
            identity=existing_user.username,
            additional_claims={
                "role": existing_user.role,
            },
        )

        refresh_token = create_refresh_token(
            identity=existing_user.username,
            additional_claims={
                "role": existing_user.role,
            },
        )

        return {
            "message": "Logged in successfully",
            "accessToken": access_token,
            "refreshToken": refresh_token,
        }, 200


# ------------------------
# CRUD Endpoints for admin
# ------------------------
@users_ns.route("")
class UserList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @users_ns.expect(user_create_model, validate=True)
    @users_ns.doc(
        description="Add a new User account to the database.\n\n**Access:** Admin only"
    )
    @users_ns.marshal_with(user_response_model, code=201)
    @users_ns.response(201, "User created successfully")
    @users_ns.response(400, "Invalid request payload")
    @users_ns.response(401, "Unauthorised access, please log in")
    @users_ns.response(403, "Insufficient permission and/or deactivated account")
    @users_ns.response(409, "Duplicate username")
    def post(self):
        """Add a new User account"""
        new_user_data = users_ns.payload

        request_username = new_user_data.get("username").strip()
        request_password = new_user_data.get("password")
        if len(request_username) < 3 or len(request_password) < 3:
            users_ns.abort(
                400, "Username and/or password must be at least 3 characters long"
            )

        if (
            db.session.execute(
                db.select(User).filter_by(username=request_username)
            ).scalar_one_or_none()
            is not None
        ):
            users_ns.abort(
                409, f"User with username {request_username} already existed"
            )

        new_user = User(
            username=request_username,
            password=hash_password(request_password),
            role=UserRole.USER,
        )
        db.session.add(new_user)
        db.session.commit()

        return new_user, 201

    # TODO: Add pagination later
    @users_ns.doc(
        description="Retrieve a list of accounts alongside with their details.\n\n**Access:** Admin only"
    )
    @users_ns.marshal_with(user_response_model, as_list=True, code=200)
    @users_ns.response(200, "List of users")
    @users_ns.response(401, "Unauthorised access, please log in")
    @users_ns.response(403, "Insufficient permission and/or deactivated account")
    def get(self):
        """Retrieve a list of users"""
        get_all_users_statement = db.select(User)
        return db.session.scalars(get_all_users_statement).all(), 200


@users_ns.route("/<int:user_id>")
@users_ns.param("user_id", "The user's unique ID")
class UserItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @users_ns.doc(
        description="Delete a user with specified ID.\n\n**Access:** Admin only"
    )
    @users_ns.response(204, "Deleted this user")
    @users_ns.response(401, "Unauthorised access, please log in")
    @users_ns.response(403, "Insufficient permission to perform this action")
    @users_ns.response(404, "User account not found")
    def delete(self, user_id):
        """Delete a specific user"""
        target_user = db.session.get(User, user_id)
        if target_user is None:
            users_ns.abort(404, "Target user to be deleted not found")

        if target_user.role == "Admin":
            users_ns.abort(403, "Cannot delete Admin account")

        db.session.delete(target_user)
        db.session.commit()

        return "", 204

    @users_ns.expect(user_update_activation_model, validate=True)
    @users_ns.doc(
        description="Activate/Deactivate a User account with a specified user_id.\n\n**Access:** Admin only"
    )
    @users_ns.marshal_with(user_response_model, code=200)
    @users_ns.response(200, "Updated successfully")
    @users_ns.response(401, "Unauthorised access, please log in")
    @users_ns.response(403, "Insufficient permission to perform this action")
    @users_ns.response(404, "User not found")
    def patch(self, user_id):
        """Update user activation status"""
        data = users_ns.payload

        target_status = data.get("active")

        target_user = db.session.get(User, user_id)
        if target_user is None:
            users_ns.abort(404, "Target user to be updated not found")

        if target_user.role == UserRole.ADMIN:
            users_ns.abort(403, "Cannot take action towards Admin account")

        if target_user.active == target_status:
            users_ns.abort(400, "Unchanged status, remain as-is")

        target_user.active = target_status
        db.session.add(target_user)
        db.session.commit()

        return target_user, 200

    # TODO: Add get later so that an admin can inspect one account


# ------------------------
# CRUD Endpoints for user
# ------------------------


# ---------------------
# Run Flask application
# ---------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1", port=5000, debug=True
    )  # Be accessible via a browser at: http://127.0.0.1:5000
