from flask import Flask
from flask_restx import Api, Resource, fields
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

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

# --------------------------------------
# Config SQLite database connection here
# --------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app.db"
db = SQLAlchemy(app)


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


# Initiate SQLite database on first start-up,
# reuse existing database i.e: app.db for subsequent run
with app.app_context():
    db.create_all()

    if not User.query.filter_by(username="admin").first():
        # Seed 1 unique admin, with 1 user on start-up (as required by the spec)
        admin = User(username="admin", password="admin", role="Admin")
        user = User(username="user", password="user", role="User")

        db.session.add(admin)
        db.session.add(user)
        db.session.commit()
        print("Automatically create 1 admin account and 1 user account.")

# ---------------------------------------
# User Model Schema for API documentation
# ---------------------------------------
user_create_model = users_ns.model(
    "UserCreate",
    {
        "username": fields.String(
            description="Account name to log into the app of this user",
            min_length=3,
            max_length=80,
        ),
        "password": fields.String(
            description="Password this user account", min_length=3, max_length=15
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


# ------------------------
# CRUD Endpoints for admin
# ------------------------
@users_ns.route("")
class UserList(Resource):
    @users_ns.expect(user_create_model, validate=True)
    @users_ns.doc(description="Add a new User account to the database")
    @users_ns.marshal_with(user_response_model, code=201)
    @users_ns.response(201, "User created successfully")
    @users_ns.abort(400, "Invalid request payload")
    @users_ns.abort(409, "Duplicate username")
    def post(self):
        """Add a new User account"""
        new_user_data = users_ns.payload

        request_username = new_user_data.get("username")
        request_password = new_user_data.get("password")
        if len(request_username.strip()) < 3 or len(request_password) < 3:
            return {
                "message": "Username and password must be at least 3 characters long"
            }, 400

        if User.query.filter_by(username=request_username).first():
            return {
                "message": f"User with username {request_username} already existed"
            }, 409

        new_user = User(
            username=request_username,
            password=request_password,
            role="User",
        )
        db.session.add(new_user)
        db.session.commit()

        return new_user, 201, {"Location": f"/users/{new_user.id}"}


@users_ns.route("/<int:user_id>")
@users_ns.param("user_id", "The user's unique ID")
class UserItem(Resource):
    @users_ns.doc(description="Delete a user with specified ID")
    @users_ns.response(204, "Deleted this user")
    @users_ns.abort(403, "Admin account cannot be deleted")
    @users_ns.abort(404, "User account not found")
    def delete(self, user_id):
        target_user = db.session.get(User, user_id)
        if target_user is None:
            return {"message": "Target user to be deleted not found"}, 404

        if target_user.role == "Admin":
            return {"message": "Cannot delete Admin account"}, 403

        db.session.delete(target_user)
        db.session.commit()

        return "", 204


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
