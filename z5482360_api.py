from flask import Flask
from flask_restx import Api, Resource, fields
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
)
import bcrypt


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
    doc="/",
)

# Construct namespace
auth_ns = api.namespace("auth", description="Login")
users_ns = api.namespace("users", description="User management (for Admin only)")


# -------------------------------------------
# Hashing user password for security purposes
# -------------------------------------------
def hash_password(pw: str):
    password_bytes = pw.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(password_bytes, salt)
    return hashed_bytes.decode("utf-8")


def check_password(pw: str, hashed_pw: str):
    password_bytes = pw.encode("utf-8")
    hashed_password_bytes = hashed_pw.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_password_bytes)


# --------------------------------------------------
# User entity definition for storing in the database
# --------------------------------------------------
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


# Initiate SQLite database on first start-up,
# reuse existing database i.e: app.db for subsequent run
with app.app_context():
    db.create_all()

    if not User.query.filter_by(username="admin").first():
        # Seed 1 unique admin, with 1 user on start-up (as required by the spec)
        admin = User(username="admin", password=hash_password("admin"), role="Admin")
        user = User(username="user", password=hash_password("user"), role="User")

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
            max_length=80,
        ),
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
    "Log In",
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
            max_length=80,
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
    @auth_ns.response(200, "Log in successfully")
    @auth_ns.response(401, "Unauthorised access")
    def post(self):
        user_data = auth_ns.payload

        existing_user = User.retrieve_user_by_username(
            username=user_data.get("username")
        )
        if existing_user is None:
            auth_ns.abort(401, "Unauthorised access")

        if not existing_user.password_check(user_data.get("password")):
            auth_ns.abort(401, "Unauthorised access")

        access_token = create_access_token(
            identity=existing_user.username,
            additional_claims={
                "role": existing_user.role,
                "active": existing_user.active,
            },
        )

        refresh_token = create_refresh_token(
            identity=existing_user.username,
            additional_claims={
                "role": existing_user.role,
                "active": existing_user.active,
            },
        )

        return {
            "message": "Logged in successfully",
            "access_token": access_token,
            "refresh_token": refresh_token,
        }, 200


# ------------------------
# CRUD Endpoints for admin
# ------------------------
@users_ns.route("")
class UserList(Resource):
    @users_ns.expect(user_create_model, validate=True)
    @users_ns.doc(description="Add a new User account to the database")
    @users_ns.marshal_with(user_response_model, code=201)
    @users_ns.response(201, "User created successfully")
    @users_ns.response(400, "Invalid request payload")
    @users_ns.response(401, "Unauthorised access, please log in")
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

        if User.query.filter_by(username=request_username).first():
            users_ns.abort(
                409, f"User with username {request_username} already existed"
            )

        new_user = User(
            username=request_username,
            password=hash_password(request_password),
            role="User",
        )
        db.session.add(new_user)
        db.session.commit()

        return new_user, 201, {"Location": f"/users/{new_user.id}"}

    @users_ns.doc(
        description="Retrieve a list of accounts alongside with their details"
    )
    @users_ns.marshal_with(user_response_model, as_list=True, code=200)
    @users_ns.response(200, "List of users")
    def get(self):
        get_all_users_statement = db.select(User)
        return db.session.scalars(get_all_users_statement).all(), 200


@users_ns.route("/<int:user_id>")
@users_ns.param("user_id", "The user's unique ID")
class UserItem(Resource):
    @users_ns.doc(description="Delete a user with specified ID")
    @users_ns.response(204, "Deleted this user")
    @users_ns.response(401, "Unauthorised access, please log in")
    @users_ns.response(403, "Admin account cannot be deleted")
    @users_ns.response(404, "User account not found")
    def delete(self, user_id):
        target_user = db.session.get(User, user_id)
        if target_user is None:
            users_ns.abort(404, "Target user to be deleted not found")

        if target_user.role == "Admin":
            users_ns.abort(403, "Cannot delete Admin account")

        db.session.delete(target_user)
        db.session.commit()

        return "", 204

    @users_ns.doc(
        description="Activate/Deactivate a User account with a specified user_id"
    )
    @users_ns.response(200, "Updated successfully")
    @users_ns.response(401, "Unauthorised access, please log in")
    @users_ns.response(403, "Admin account cannot be deactivated")
    @users_ns.response(404, "User not found")
    def patch(self, user_id):
        target_user = db.session.get(User, user_id)
        if target_user is None:
            users_ns.abort(404, "Target user to be updated not found")

        if target_user.role == "Admin":
            users_ns.abort(403, "Cannot deactivate Admin account")

        new_activation_status = not target_user.active
        target_user.active = new_activation_status
        db.session.add(target_user)
        db.session.commit()

        return {"message": "Updated successfully"}, 200


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
