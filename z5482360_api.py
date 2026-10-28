import os
from enum import StrEnum
from functools import wraps


import bcrypt
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, request
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
from loguru import logger
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

app = Flask(__name__)

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
admin_ns = api.namespace(
    "admin", description="User management and importing files (Admin access only)"
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
    longtitude = db.Column(db.Float, nullable=False)
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
user_create_model = admin_ns.model(
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

user_update_activation_model = admin_ns.model(
    "UserActivationUpdate",
    {
        "active": fields.Boolean(
            description="Target status that admin wants to update", required=True
        )
    },
    strict=True,
)

user_response_model = admin_ns.model(
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


# ----------------------------------------------------
# Requirement Set 1: Admin CRUD/RESTful API endpoints
# ----------------------------------------------------
@admin_ns.route("/user")
class UserList(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @admin_ns.expect(user_create_model, validate=True)
    @admin_ns.doc(
        description="Add a new User account to the database.\n\n**Access:** Admin only"
    )
    @admin_ns.marshal_with(user_response_model, code=201)
    @admin_ns.response(201, "User created successfully")
    @admin_ns.response(400, "Invalid request payload")
    @admin_ns.response(401, "Unauthorised access, please log in")
    @admin_ns.response(403, "Insufficient permission and/or deactivated account")
    @admin_ns.response(409, "Duplicate username")
    def post(self):
        """Add a new User account"""
        new_user_data = admin_ns.payload

        request_username = new_user_data.get("username").strip()
        request_password = new_user_data.get("password")
        if len(request_username) < 3 or len(request_password) < 3:
            admin_ns.abort(
                400, "Username and/or password must be at least 3 characters long"
            )

        if (
            db.session.execute(
                db.select(User).filter_by(username=request_username)
            ).scalar_one_or_none()
            is not None
        ):
            admin_ns.abort(
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
    @admin_ns.doc(
        description="Retrieve a list of accounts alongside with their details.\n\n**Access:** Admin only"
    )
    @admin_ns.marshal_with(user_response_model, as_list=True, code=200)
    @admin_ns.response(200, "List of users")
    @admin_ns.response(401, "Unauthorised access, please log in")
    @admin_ns.response(403, "Insufficient permission and/or deactivated account")
    def get(self):
        """Retrieve a list of users"""
        get_all_users_statement = db.select(User)
        return db.session.scalars(get_all_users_statement).all(), 200


@admin_ns.route("/user/<int:user_id>")
@admin_ns.param("user_id", "The user's unique ID")
class UserItem(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @admin_ns.doc(
        description="Delete a user with specified ID.\n\n**Access:** Admin only"
    )
    @admin_ns.response(204, "Deleted this user")
    @admin_ns.response(401, "Unauthorised access, please log in")
    @admin_ns.response(403, "Insufficient permission to perform this action")
    @admin_ns.response(404, "User account not found")
    def delete(self, user_id):
        """Delete a specific user"""
        target_user = db.session.get(User, user_id)
        if target_user is None:
            admin_ns.abort(404, "Target user to be deleted not found")

        if target_user.role == "Admin":
            admin_ns.abort(403, "Cannot delete Admin account")

        db.session.delete(target_user)
        db.session.commit()

        return "", 204

    @admin_ns.expect(user_update_activation_model, validate=True)
    @admin_ns.doc(
        description="Activate/Deactivate a User account with a specified user_id.\n\n**Access:** Admin only"
    )
    @admin_ns.marshal_with(user_response_model, code=200)
    @admin_ns.response(200, "Updated successfully")
    @admin_ns.response(401, "Unauthorised access, please log in")
    @admin_ns.response(403, "Insufficient permission to perform this action")
    @admin_ns.response(404, "User not found")
    def patch(self, user_id):
        """Update user activation status"""
        data = admin_ns.payload

        target_status = data.get("active")

        target_user = db.session.get(User, user_id)
        if target_user is None:
            admin_ns.abort(404, "Target user to be updated not found")

        if target_user.role == UserRole.ADMIN:
            admin_ns.abort(403, "Cannot take action towards Admin account")

        if target_user.active == target_status:
            admin_ns.abort(400, "Unchanged status, remain as-is")

        target_user.active = target_status
        db.session.add(target_user)
        db.session.commit()

        return target_user, 200

    # TODO: Add get later so that an admin can inspect one account


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


class ImportValidationError(Exception):
    """Raised for any problem with an upload; carries the HTTP status to return."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def dataset_name_matcher(field, file):
    """Work out which dataset a file is: by form-field name first, then by file name."""
    for candidate in (field, file.filename):
        file_name = (candidate or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        file_name = file_name.removesuffix(".csv")

        if file_name in MODELS:
            return file_name

        raise ImportValidationError(
            f"Cannot tell which dataset '{file.filename}' (field '{field}') is. "
            f"Use form fields / file names: {', '.join(m + '.csv' for m in MODELS)}"
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
        return dict(zip(frames[dataset][id_col]), zip(frames[dataset][iata_col]))

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
        ref = lookup(frames, dataset, id_col, iata_col)

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
                db.session.execute(db.session.delete(MODELS[name]))

        for name in ("airlines", "airports", "routes"):
            if name in frames:
                records = [
                    {k: (None if v == "" else v) for k, v in rows.item()}
                    for rows in frames[name].to_dict("records")
                ]
                db.session.execute(db.insert(MODELS[name]), records)

        db.session.commit()
    except Exception:
        db.session.rollback()
        raise


def check_travel_plans(frames):
    """Requirement Set 4 hook: decide what happens to plans that use replaced routes.

    TODO: once travel plans exist, compare the plans' route ids with
    routes_view(frames)["route_id"] and either raise ImportValidationError(..., 409)
    or mark the affected plans. Document the chosen policy in the upload endpoint.
    """


@admin_ns.route("/upload")
class FileUpload(Resource):
    method_decorators = [require_roles(UserRole.ADMIN)]

    @admin_ns.doc(
        description=(
            "Import any subset of `airports.csv`, `airlines.csv`, `routes.csv` "
            "(multipart/form-data; form fields named `airports`, `airlines`, `routes`, "
            "or any field name as long as the file is called airports.csv / airlines.csv / routes.csv).\n\n"
            "Each uploaded file **replaces all existing records** of its dataset; datasets not "
            "in the request are untouched. Every file is validated (columns, types, unique ids, "
            "id/iata consistency) before anything is written, referential integrity is checked "
            "on the combined result of the request, and the import is atomic.\n\n"
            "**Existing travel plans:** TODO - document the chosen policy here.\n\n"
            "**Access:** Admin only"
        )
    )
    @admin_ns.response(200, "Import successful")
    @admin_ns.response(
        400, "No files / malformed file / schema or consistency violation"
    )
    @admin_ns.response(401, "Unauthorised access, please log in")
    @admin_ns.response(403, "Insufficient permission and/or deactivated account")
    @admin_ns.response(409, "Referential integrity (or travel plan) conflict")
    def post(self):
        """Import airports, airlines and/or routes from CSV files"""
        try:
            uploads = {}

            for field, file in request.files.items(multi=True):
                if not file.filename:
                    continue

                name = dataset_name_matcher(field, file)
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
            admin_ns.abort(e.status, e.message)

        try:
            replace_data(frames)
        except SQLAlchemyError:
            logger.exception("CSV import failed, transaction rolled back")
            admin_ns.abort(500, "Import failed, no changes were made")

        logger.info("Imported datasets: {}", {n: len(df) for n, df in frames.items()})
        return {
            "message": "Import successful",
            "imported": {name: len(df) for name, df in frames.items()},
        }, 200


# ---------------------
# Run Flask application
# ---------------------
if __name__ == "__main__":
    app.run(
        host="127.0.0.1", port=5000, debug=True
    )  # Be accessible via a browser at: http://127.0.0.1:5000
