"""
Automated REST-client test suite for z5482360_api.py (COMP9321 Assignment 2, Requirement Set 6).

How to run
----------
1. Start the API with a freshly created database. To keep your normal database untouched,
   point the API at a throw-away test database through the DATABASE_URL environment variable
   (Flask-SQLAlchemy puts relative SQLite paths in ./instance/, so this creates instance/test.db;
   delete that file to start from scratch again):

       DATABASE_URL=sqlite:///test.db python z5482360_api.py

2. With airports.csv, airlines.csv and routes.csv in the working directory, run:

       pytest z5482360_tests.py

   Set API_BASE_URL to test a server somewhere other than http://127.0.0.1:5000/api/v1.

Design of the suite (why the tests are independent of each other)
------------------------------------------------------------------
* A session fixture logs in as the seeded `admin` account and imports the three CSV files once,
  so every test starts from the same airports/airlines/routes data.
* Every test that needs an account creates its own throw-away User with a random username
  (the `new_user` fixture) and deletes it afterwards. Deleting a user also deletes their travel
  plans, so tests never see each other's plans, never depend on execution order, and the suite
  can be re-run against the same database.
* The seeded `user` account is only ever read (logged in), never deactivated or deleted.
* Tests that change the imported data successfully (only one does) restore the original CSV
  files in a `finally` block. All other import tests send invalid files, which the API must
  reject atomically, so they leave the data untouched (and assert exactly that).
* Expected values are computed from the CSV files themselves (row counts, filters), so the
  tests check data correctness rather than hard-coded numbers.

Each test's docstring states the requirement number it covers and its purpose.
"""

import csv
import io
import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000/api/v1").rstrip("/")
TIMEOUT = 30
DATASETS = ("airports", "airlines", "routes")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def csv_dir():
    """The spec puts the CSV files in the working directory; fall back to this file's folder."""
    for folder in (Path.cwd(), Path(__file__).resolve().parent):
        if all((folder / f"{name}.csv").exists() for name in DATASETS):
            return folder
    pytest.exit("airports.csv, airlines.csv and routes.csv must be in the working directory")


def read_csv_bytes(name):
    return (csv_dir() / f"{name}.csv").read_bytes()


def read_csv_rows(name):
    with open(csv_dir() / f"{name}.csv", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def call(method, path, token=None, **kwargs):
    """Send one request to the API, optionally with a Bearer token."""
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return requests.request(
        method, f"{BASE_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs
    )


def follow(link, token):
    """Follow a link returned by the API (links are absolute paths such as /api/v1/routes/1)."""
    root = BASE_URL.split("/api/")[0]
    return requests.get(
        f"{root}{link}", headers={"Authorization": f"Bearer {token}"}, timeout=TIMEOUT
    )


def login(username, password):
    response = call("POST", "/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["accessToken"]


def unique(prefix):
    """A username/plan name that cannot clash with data left over from earlier runs."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def csv_file(name, content=None, filename=None):
    """One multipart file part: (form field, (file name, bytes, content type))."""
    data = read_csv_bytes(name) if content is None else content
    if isinstance(data, str):
        data = data.encode("utf-8")
    return (name, (filename or f"{name}.csv", data, "text/csv"))


def edit_csv(name, edit_row=None, keep_row=None, extra_rows=()):
    """Return a modified copy of a CSV file as text.

    edit_row(row) may change a row dict in place; keep_row(row) -> False drops the row;
    extra_rows are appended as dicts.
    """
    rows = read_csv_rows(name)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if keep_row and not keep_row(row):
            continue
        if edit_row:
            edit_row(row)
        writer.writerow(row)
    for row in extra_rows:
        writer.writerow(row)
    return out.getvalue()


def import_files(token, *parts):
    return call("POST", "/imports", token, files=list(parts))


def days_from_today(days):
    return (date.today() + timedelta(days=days)).isoformat()


def plan_payload(**overrides):
    payload = {
        "name": unique("plan"),
        "departure_date": days_from_today(30),
        "return_date": days_from_today(35),
    }
    payload.update(overrides)
    return payload


def create_plan(token, **overrides):
    response = call("POST", "/travel-plans", token, json=plan_payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def assert_validation_error(response, field):
    """flask-restx validation errors look like {"message": ..., "errors": {key: text}}.

    The key is not always the field name (a bad list item is keyed "route_ids.0", an unknown
    property is keyed ""), so the field is looked for anywhere in the errors.
    """
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["message"] == "Input payload validation failed"
    assert field in json.dumps(body["errors"])


PAGE_KEYS = {"page", "per_page", "total", "pages"}


def assert_page(body, key):
    """Every list endpoint shares the same page envelope."""
    assert set(body) == PAGE_KEYS | {key}
    assert isinstance(body[key], list)
    assert body["per_page"] >= len(body[key])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def admin_token():
    return login("admin", "admin")


@pytest.fixture(scope="session", autouse=True)
def imported_data(admin_token):
    """Import the three provided CSV files once, so every test sees the same dataset."""
    response = import_files(admin_token, *(csv_file(name) for name in DATASETS))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="session")
def data():
    """The CSV contents, used to compute expected results."""
    return SimpleNamespace(**{name: read_csv_rows(name) for name in DATASETS})


@pytest.fixture
def new_user(admin_token):
    """Factory: create a fresh User account (random name) and delete it after the test."""
    created = []

    def make():
        username, password = unique("tester"), "secret123"
        response = call(
            "POST", "/users", admin_token, json={"username": username, "password": password}
        )
        assert response.status_code == 201, response.text
        account = SimpleNamespace(**response.json(), password=password)
        account.token = login(username, password)
        created.append(account.id)
        return account

    yield make
    for user_id in created:
        call("DELETE", f"/users/{user_id}", admin_token)


@pytest.fixture
def user(new_user):
    """A single fresh User account for tests that only need one."""
    return new_user()


@pytest.fixture
def admin_plans(admin_token):
    """Tracks plans the admin creates so they are removed after the test."""
    created = []
    yield created
    for plan_id in created:
        call("DELETE", f"/travel-plans/{plan_id}", admin_token)


# ===========================================================================
# General: Swagger UI and documentation
# ===========================================================================
class TestSwagger:
    def test_swagger_ui_served_at_root(self):
        """General - Purpose: the Swagger UI page is reachable at http://127.0.0.1:5000/."""
        root = BASE_URL.split("/api/")[0]
        response = requests.get(f"{root}/", timeout=TIMEOUT)
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_swagger_json_lists_every_resource(self):
        """General - Purpose: the generated OpenAPI document contains all endpoints."""
        response = call("GET", "/swagger.json")
        assert response.status_code == 200
        paths = response.json()["paths"]
        for path in (
            "/auth/login",
            "/users",
            "/users/{user_id}",
            "/imports",
            "/airports",
            "/airports/{airport_id}",
            "/airports/{airport_id}/airlines",
            "/airlines",
            "/airlines/{airline_id}",
            "/routes",
            "/routes/{route_id}",
            "/travel-plans",
            "/travel-plans/{plan_id}",
            "/travel-plans/{plan_id}/routes",
            "/travel-plans/{plan_id}/routes/{route_id}",
            "/export/travel-plans",
        ):
            assert path in paths


# ===========================================================================
# Requirement Set 1: User management and access control
# ===========================================================================
class TestRequirement1Authentication:
    def test_login_admin_returns_token(self):
        """Requirement 1 - Purpose: the seeded admin/admin account can log in and gets a JWT."""
        response = call("POST", "/auth/login", json={"username": "admin", "password": "admin"})
        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Logged in successfully"
        assert body["accessToken"].count(".") == 2  # header.payload.signature

    def test_login_seeded_user_account(self):
        """Requirement 1 - Purpose: the seeded user/user account exists and can log in."""
        token = login("user", "user")
        assert token

    def test_login_wrong_password(self):
        """Requirement 1 - Purpose: a wrong password is rejected with 401 and a generic message."""
        response = call("POST", "/auth/login", json={"username": "admin", "password": "nope"})
        assert response.status_code == 401
        assert response.json()["message"] == "Invalid username or password"

    def test_login_unknown_user_same_message(self):
        """Requirement 1 - Purpose: an unknown username gets the same 401 message as a wrong
        password, so the API does not reveal which usernames exist."""
        response = call(
            "POST", "/auth/login", json={"username": unique("ghost"), "password": "whatever"}
        )
        assert response.status_code == 401
        assert response.json()["message"] == "Invalid username or password"

    def test_login_missing_field(self):
        """Requirement 1 - Purpose: a login payload without a password fails validation (400)."""
        response = call("POST", "/auth/login", json={"username": "admin"})
        assert_validation_error(response, "password")

    def test_request_without_token(self):
        """Requirement 1 - Purpose: protected endpoints reject requests without a token (401)."""
        response = call("GET", "/airports")
        assert response.status_code == 401
        assert "Missing Authorization Header" in response.json()["message"]

    def test_request_with_malformed_token(self):
        """Requirement 1 - Purpose: a malformed token is rejected with 401 (not 422/500)."""
        response = call("GET", "/airports", "not-a-real-token")
        assert response.status_code == 401
        assert response.json()["message"].startswith("Invalid token")

    def test_request_with_tampered_token(self, admin_token):
        """Requirement 1 - Purpose: a token whose signature was altered is rejected with 401."""
        header, payload, signature = admin_token.split(".")
        tampered = f"{header}.{payload}.{signature[::-1]}"
        response = call("GET", "/users", tampered)
        assert response.status_code == 401
        assert response.json()["message"].startswith("Invalid token")


class TestRequirement1UserManagement:
    def test_admin_creates_user(self, admin_token):
        """Requirement 1 - Purpose: the admin creates a User account; the response describes the
        account and never includes the password."""
        username = unique("created")
        response = call(
            "POST", "/users", admin_token, json={"username": username, "password": "pw1234"}
        )
        assert response.status_code == 201
        body = response.json()
        try:
            assert set(body) == {"id", "username", "role", "active"}
            assert body["username"] == username
            assert body["role"] == "User"
            assert body["active"] is True
        finally:
            call("DELETE", f"/users/{body['id']}", admin_token)

    def test_create_duplicate_username(self, admin_token, user):
        """Requirement 1 - Purpose: usernames are unique; a duplicate is rejected with 409."""
        response = call(
            "POST", "/users", admin_token, json={"username": user.username, "password": "abc123"}
        )
        assert response.status_code == 409
        assert "already existed" in response.json()["message"]

    def test_create_user_too_short(self, admin_token):
        """Requirement 1 - Purpose: usernames/passwords shorter than 3 characters fail validation."""
        response = call("POST", "/users", admin_token, json={"username": "ab", "password": "abc"})
        assert_validation_error(response, "username")

    def test_create_user_whitespace_padded_name(self, admin_token):
        """Requirement 1 - Purpose: a name that only reaches 3 characters with surrounding spaces
        is rejected after trimming."""
        response = call(
            "POST", "/users", admin_token, json={"username": "  ab  ", "password": "abc123"}
        )
        assert response.status_code == 400
        assert "at least 3 characters" in response.json()["message"]

    def test_create_user_rejects_unknown_fields(self, admin_token):
        """Requirement 1 - Purpose: the payload is strict, so a client cannot set e.g. the role."""
        response = call(
            "POST",
            "/users",
            admin_token,
            json={"username": unique("x"), "password": "abc123", "role": "Admin"},
        )
        assert response.status_code == 400
        assert "role" in str(response.json()["errors"])

    def test_list_users_paginated_without_passwords(self, admin_token, new_user):
        """Requirement 1 - Purpose: the admin lists accounts page by page, ordered by id, and no
        credentials are ever returned."""
        new_user(), new_user()
        response = call("GET", "/users?page=1&per_page=2", admin_token)
        assert response.status_code == 200
        body = response.json()
        assert_page(body, "users")
        assert body["page"] == 1 and body["per_page"] == 2
        assert len(body["users"]) == 2
        assert body["total"] >= 4  # admin, user and the two just created
        assert body["users"][0]["username"] == "admin"  # first account created
        for account in body["users"]:
            assert set(account) == {"id", "username", "role", "active"}
            assert "password" not in str(account).lower()

    def test_list_users_page_past_end(self, admin_token):
        """Requirement 1 - Purpose: a page past the end is an empty list, not an error."""
        response = call("GET", "/users?page=100000", admin_token)
        assert response.status_code == 200
        assert response.json()["users"] == []

    @pytest.mark.parametrize("query", ["page=0", "per_page=0", "per_page=101", "page=abc"])
    def test_list_users_invalid_pagination(self, admin_token, query):
        """Requirement 1 - Purpose: invalid page/per_page values are rejected with 400."""
        response = call("GET", f"/users?{query}", admin_token)
        assert_validation_error(response, query.split("=")[0])

    def test_get_single_user(self, admin_token, user):
        """Requirement 1 - Purpose: the admin retrieves one account by id."""
        response = call("GET", f"/users/{user.id}", admin_token)
        assert response.status_code == 200
        assert response.json() == {
            "id": user.id,
            "username": user.username,
            "role": "User",
            "active": True,
        }

    def test_get_unknown_user(self, admin_token):
        """Requirement 1 - Purpose: an unknown user id returns 404 with a clear message."""
        response = call("GET", "/users/99999999", admin_token)
        assert response.status_code == 404
        assert response.json()["message"] == "User 99999999 not found"

    def test_user_cannot_manage_users(self, user):
        """Requirement 1 - Purpose: a normal User cannot list, create, read or delete accounts."""
        for method, path, payload in (
            ("GET", "/users", None),
            ("POST", "/users", {"username": unique("x"), "password": "abc123"}),
            ("GET", f"/users/{user.id}", None),
            ("DELETE", f"/users/{user.id}", None),
            ("PATCH", f"/users/{user.id}", {"active": False}),
        ):
            response = call(method, path, user.token, json=payload)
            assert response.status_code == 403, (method, path)
            assert response.json()["message"] == "Insufficient permission to perform this action"

    def test_deactivate_blocks_access_and_reactivate_restores_it(self, admin_token, user):
        """Requirement 1 - Purpose: a deactivated account loses API access immediately (even
        with a token issued earlier) and cannot log in; reactivating restores access."""
        response = call("PATCH", f"/users/{user.id}", admin_token, json={"active": False})
        assert response.status_code == 200
        assert response.json()["active"] is False

        response = call("GET", "/airports", user.token)
        assert response.status_code == 403
        assert response.json()["message"] == "This account has been deactivated"

        response = call(
            "POST", "/auth/login", json={"username": user.username, "password": user.password}
        )
        assert response.status_code == 403
        assert response.json()["message"] == "Deactivated account cannot access internal service"

        response = call("PATCH", f"/users/{user.id}", admin_token, json={"active": True})
        assert response.status_code == 200
        assert call("GET", "/airports", user.token).status_code == 200

    def test_patch_same_status_is_idempotent(self, admin_token, user):
        """Requirement 1 - Purpose: setting the status an account already has succeeds (200)."""
        for _ in range(2):
            response = call("PATCH", f"/users/{user.id}", admin_token, json={"active": True})
            assert response.status_code == 200
            assert response.json()["active"] is True

    def test_patch_requires_boolean(self, admin_token, user):
        """Requirement 1 - Purpose: the activation payload must be a boolean."""
        response = call("PATCH", f"/users/{user.id}", admin_token, json={"active": "no"})
        assert_validation_error(response, "active")

    def test_admin_cannot_be_deactivated_or_deleted(self, admin_token):
        """Requirement 1 - Purpose: the admin account itself can be neither deactivated nor
        deleted."""
        admin_id = call("GET", "/users?per_page=1", admin_token).json()["users"][0]["id"]

        response = call("PATCH", f"/users/{admin_id}", admin_token, json={"active": False})
        assert response.status_code == 403
        assert response.json()["message"] == "Cannot take action towards Admin account"

        response = call("DELETE", f"/users/{admin_id}", admin_token)
        assert response.status_code == 403
        assert response.json()["message"] == "Cannot delete Admin account"

        assert call("GET", "/users", admin_token).status_code == 200  # still active

    def test_delete_user_revokes_access(self, admin_token, user):
        """Requirement 1 - Purpose: deleting an account removes it and its old token stops
        working (401)."""
        response = call("DELETE", f"/users/{user.id}", admin_token)
        assert response.status_code == 204
        assert response.text == ""

        assert call("GET", f"/users/{user.id}", admin_token).status_code == 404
        response = call("GET", "/airports", user.token)
        assert response.status_code == 401
        assert response.json()["message"] == "Unauthorised access, please log in"

    def test_delete_unknown_user(self, admin_token):
        """Requirement 1 - Purpose: deleting a non-existent account returns 404."""
        response = call("DELETE", "/users/99999999", admin_token)
        assert response.status_code == 404
        assert response.json()["message"] == "Target user to be deleted not found"

    def test_deleted_id_is_never_reused(self, admin_token, new_user):
        """Requirement 1 - Purpose: a new account never gets the id of a deleted one, so an old
        token cannot be used to act as the new account."""
        first = new_user()
        call("DELETE", f"/users/{first.id}", admin_token)
        second = new_user()
        assert second.id > first.id
        assert call("GET", "/airports", first.token).status_code == 401


# ===========================================================================
# Requirement Set 2: Importing airports, airlines and routes
# ===========================================================================
class TestRequirement2Import:
    def test_import_all_three_files(self, admin_token, data):
        """Requirement 2 - Purpose: importing all three CSV files replaces the data and reports
        the number of records per dataset (matching the files)."""
        response = import_files(admin_token, *(csv_file(name) for name in DATASETS))
        assert response.status_code == 200
        body = response.json()
        assert body["message"] == "Import successful"
        assert body["imported"] == {
            "airports": len(data.airports),
            "airlines": len(data.airlines),
            "routes": len(data.routes),
        }

    def test_import_single_dataset(self, admin_token, data):
        """Requirement 2 - Purpose: a single file can be imported on its own; only that dataset
        is reported."""
        response = import_files(admin_token, csv_file("airports"))
        assert response.status_code == 200
        assert response.json()["imported"] == {"airports": len(data.airports)}

    def test_import_any_field_name(self, admin_token, data):
        """Requirement 2 - Purpose: the dataset is recognised by the file name, whatever the form
        field is called."""
        response = call(
            "POST",
            "/imports",
            admin_token,
            files=[("upload", ("airlines.csv", read_csv_bytes("airlines"), "text/csv"))],
        )
        assert response.status_code == 200
        assert response.json()["imported"] == {"airlines": len(data.airlines)}

    def test_import_forbidden_for_users(self, user):
        """Requirement 2 - Purpose: only the admin may import data (User gets 403)."""
        response = import_files(user.token, csv_file("airports"))
        assert response.status_code == 403
        assert response.json()["message"] == "Insufficient permission to perform this action"

    def test_import_without_token(self):
        """Requirement 2 - Purpose: an unauthenticated import is rejected with 401."""
        response = import_files(None, csv_file("airports"))
        assert response.status_code == 401

    def test_import_without_files(self, admin_token):
        """Requirement 2 - Purpose: a request with no files is a 400."""
        response = call("POST", "/imports", admin_token)
        assert response.status_code == 400
        assert response.json()["message"] == "No files uploaded"

    def test_import_unexpected_file_name(self, admin_token):
        """Requirement 2 - Purpose: files must be named exactly like the provided ones."""
        response = import_files(admin_token, csv_file("airports", filename="Airports.csv"))
        assert response.status_code == 400
        assert "Unexpected file name 'Airports.csv'" in response.json()["message"]

    def test_import_same_dataset_twice(self, admin_token):
        """Requirement 2 - Purpose: uploading the same dataset twice in one request is ambiguous
        and rejected."""
        response = import_files(admin_token, csv_file("airlines"), csv_file("airlines"))
        assert response.status_code == 400
        assert response.json()["message"] == "airlines was uploaded more than once"

    @pytest.mark.parametrize(
        "name, content, expected",
        [
            pytest.param("airlines", "", "not a readable CSV file", id="empty-file"),
            pytest.param(
                "airlines",
                "airline_id,name,iata_code,icao_code,country\n",
                "contains no data rows",
                id="header-only",
            ),
            pytest.param(
                "airlines",
                "airline_id,name,iata,icao_code,country\n1,Qantas,QF,QFA,Australia\n",
                "has wrong columns",
                id="renamed-column",
            ),
            pytest.param(
                "airlines",
                "airline_id,name,iata_code,icao_code,country,extra\n1,Q,QF,QFA,AU,x\n",
                "Unexpected: ['extra']",
                id="extra-column",
            ),
            pytest.param(
                "airlines",
                "airline_id,name,iata_code,icao_code,country\none,Qantas,QF,QFA,Australia\n",
                "'airline_id' must be int",
                id="non-integer-id",
            ),
            pytest.param(
                "airlines",
                "airline_id,name,iata_code,icao_code,country\n1,,QF,QFA,Australia\n",
                "'name' is empty at line 2",
                id="blank-required-value",
            ),
            pytest.param(
                "airlines",
                "airline_id,name,iata_code,icao_code,country\n"
                "1,Qantas,QF,QFA,Australia\n1,Other,ZZ,ZZZ,Nowhere\n",
                "duplicate airline_id '1' at line 3",
                id="duplicate-id",
            ),
            pytest.param(
                "airlines",
                "airline_id,name,iata_code,icao_code,country\n"
                "1,Qantas,QF,QFA,Australia\n2,Other,QF,ZZZ,Nowhere\n",
                "duplicate iata_code 'QF'",
                id="duplicate-iata",
            ),
        ],
    )
    def test_import_rejects_invalid_airlines_file(
        self, admin_token, data, name, content, expected
    ):
        """Requirement 2 - Purpose: malformed files (empty, wrong columns, wrong types, blanks,
        duplicate identifiers) are rejected with 400 and a message naming the problem, and the
        stored data is unchanged."""
        response = import_files(admin_token, csv_file(name, content))
        assert response.status_code == 400
        assert expected in response.json()["message"]
        total = call("GET", "/airlines", admin_token).json()["total"]
        assert total == len(data.airlines)

    def test_import_rejects_out_of_range_latitude(self, admin_token):
        """Requirement 2 - Purpose: values outside a valid range (latitude 95) are rejected."""
        content = edit_csv(
            "airports", edit_row=lambda r: r.update(latitude="95") if r["airport_id"] == "1" else None
        )
        response = import_files(admin_token, csv_file("airports", content))
        assert response.status_code == 400
        assert "'latitude' must be between -90 and 90" in response.json()["message"]

    def test_import_rejects_non_numeric_coordinate(self, admin_token):
        """Requirement 2 - Purpose: a float column containing text is rejected."""
        content = edit_csv(
            "airports",
            edit_row=lambda r: r.update(longitude="east") if r["airport_id"] == "2" else None,
        )
        response = import_files(admin_token, csv_file("airports", content))
        assert response.status_code == 400
        assert "'longitude' must be float" in response.json()["message"]

    def test_import_rejects_inconsistent_route_iata(self, admin_token):
        """Requirement 2 - Purpose: a route whose airport id and IATA code refer to different
        airports is rejected (consistency of related fields)."""
        content = edit_csv(
            "routes",
            edit_row=lambda r: r.update(origin_iata="MEL") if r["route_id"] == "1" else None,
        )
        response = import_files(admin_token, csv_file("routes", content))
        assert response.status_code == 400
        message = response.json()["message"]
        assert "inconsistent origin_airport_id/origin_iata" in message
        assert "route_id 1" in message

    def test_import_rejects_orphan_routes(self, admin_token):
        """Requirement 2 - Purpose: removing an airline that routes still use breaks referential
        integrity, so the import is rejected with 409 and nothing changes."""
        content = edit_csv("airlines", keep_row=lambda r: r["iata_code"] != "JL")
        response = import_files(admin_token, csv_file("airlines", content))
        assert response.status_code == 409
        assert "would reference a non-existent airline" in response.json()["message"]
        assert call("GET", "/airlines?code=JL", admin_token).json()["total"] == 1

    def test_import_is_atomic(self, admin_token, data):
        """Requirement 2 - Purpose: when one file in a request is invalid, the valid files in
        the same request are not applied either."""
        renamed = edit_csv(
            "airports",
            edit_row=lambda r: r.update(name="ATOMICITY CHECK") if r["airport_id"] == "1" else None,
        )
        broken_airlines = "airline_id,name\n1,Qantas\n"
        response = import_files(
            admin_token, csv_file("airports", renamed), csv_file("airlines", broken_airlines)
        )
        assert response.status_code == 400
        airport = call("GET", "/airports/1", admin_token).json()
        assert airport["name"] == data.airports[0]["name"]

    def test_integrity_checked_against_combined_request(self, admin_token, data):
        """Requirement 2 - Purpose: integrity is checked on the combined result of the request:
        a route to a new airline is rejected alone (409) but accepted when the same request also
        uploads that airline. The original data is restored afterwards."""
        new_airline = {
            "airline_id": "999",
            "name": "Test Air",
            "iata_code": "T9",
            "icao_code": "TNN",
            "country": "Australia",
        }
        new_route = {
            "route_id": "99999",
            "airline_id": "999",
            "airline_iata": "T9",
            "origin_airport_id": "1",
            "origin_iata": "SYD",
            "destination_airport_id": "2",
            "destination_iata": "MEL",
            "codeshare": "",
            "stops": "0",
            "equipment": "A320",
        }
        routes = edit_csv("routes", extra_rows=[new_route])
        airlines = edit_csv("airlines", extra_rows=[new_airline])

        alone = import_files(admin_token, csv_file("routes", routes))
        assert alone.status_code == 409
        assert "non-existent airline" in alone.json()["message"]

        try:
            together = import_files(
                admin_token, csv_file("airlines", airlines), csv_file("routes", routes)
            )
            assert together.status_code == 200
            assert together.json()["imported"] == {
                "airlines": len(data.airlines) + 1,
                "routes": len(data.routes) + 1,
            }
            route = call("GET", "/routes/99999", admin_token).json()
            assert route["airline"]["name"] == "Test Air"
        finally:
            restored = import_files(admin_token, *(csv_file(name) for name in DATASETS))
            assert restored.status_code == 200
        assert call("GET", "/routes/99999", admin_token).status_code == 404

    def test_import_blocked_by_travel_plan(self, admin_token, user):
        """Requirement 2 + 4 - Purpose: a routes import that would remove a route used by a
        travel plan is rejected with 409, naming the missing route ids and the affected account
        (but not the plan's contents), and the routes are kept."""
        plan = create_plan(user.token, route_ids=[1])
        content = edit_csv("routes", keep_row=lambda r: r["route_id"] != "1")

        response = import_files(admin_token, csv_file("routes", content))
        assert response.status_code == 409
        body = response.json()
        assert body["missing_route_ids"] == [1]
        assert {"username": user.username, "plans": 1} in body["affected_accounts"]
        assert plan["name"] not in str(body)  # plan contents stay private
        assert call("GET", "/routes/1", admin_token).status_code == 200

    def test_airline_only_import_not_blocked_by_plans(self, admin_token, user):
        """Requirement 2 + 4 - Purpose: imports without routes.csv cannot remove routes, so
        travel plans never block them."""
        create_plan(user.token, route_ids=[1])
        response = import_files(admin_token, csv_file("airlines"))
        assert response.status_code == 200


# ===========================================================================
# Requirement Set 3: Exploring airports, airlines and routes
# ===========================================================================
class TestRequirement3Airports:
    def test_list_airports_page_structure(self, user, data):
        """Requirement 3 - Purpose: airports are listed page by page with totals that match the
        imported file."""
        response = call("GET", "/airports?per_page=5", user.token)
        assert response.status_code == 200
        body = response.json()
        assert_page(body, "airports")
        assert body["total"] == len(data.airports)
        assert body["pages"] == -(-len(data.airports) // 5)  # ceiling division
        assert [a["airport_id"] for a in body["airports"]] == [1, 2, 3, 4, 5]

    def test_admin_can_also_browse(self, admin_token, data):
        """Requirement 3 - Purpose: the admin can act as a normal user and browse data."""
        response = call("GET", "/airports", admin_token)
        assert response.status_code == 200
        assert response.json()["total"] == len(data.airports)

    def test_get_airport_matches_csv(self, user, data):
        """Requirement 3 - Purpose: one airport's details match its CSV row, plus navigation links
        to related resources."""
        expected = data.airports[0]
        response = call("GET", f"/airports/{expected['airport_id']}", user.token)
        assert response.status_code == 200
        airport = response.json()
        for key in ("name", "city", "country", "iata_code", "icao_code", "timezone"):
            assert airport[key] == expected[key]
        assert airport["latitude"] == pytest.approx(float(expected["latitude"]))
        assert airport["longitude"] == pytest.approx(float(expected["longitude"]))
        assert airport["altitude"] == int(expected["altitude"])
        assert set(airport["_links"]) == {
            "self",
            "airlines",
            "departing_routes",
            "arriving_routes",
        }

    def test_get_unknown_airport(self, user):
        """Requirement 3 - Purpose: an unknown airport id returns 404 with a message."""
        response = call("GET", "/airports/99999", user.token)
        assert response.status_code == 404
        assert response.json()["message"] == "Airport 99999 not found"

    def test_filter_country_partial_case_insensitive(self, user, data):
        """Requirement 3 - Purpose: `country` is a case-insensitive partial match: `AUS` finds
        both Australia and Austria."""
        response = call("GET", "/airports?country=AUS&per_page=100", user.token)
        body = response.json()
        expected = [a for a in data.airports if "aus" in a["country"].lower()]
        assert body["total"] == len(expected)
        assert {a["country"] for a in body["airports"]} == {"Australia", "Austria"}

    def test_filter_name_and_city(self, user):
        """Requirement 3 - Purpose: name and city filters match partially and combine with AND."""
        response = call("GET", "/airports?name=KINGSFORD", user.token)
        assert [a["iata_code"] for a in response.json()["airports"]] == ["SYD"]

        response = call("GET", "/airports?country=australia&city=melb", user.token)
        assert [a["iata_code"] for a in response.json()["airports"]] == ["MEL"]

    @pytest.mark.parametrize("code", ["syd", "YSSY"])
    def test_filter_code_iata_or_icao(self, user, code):
        """Requirement 3 - Purpose: `code` matches either the IATA or ICAO code exactly,
        ignoring case."""
        response = call("GET", f"/airports?code={code}", user.token)
        airports = response.json()["airports"]
        assert [a["iata_code"] for a in airports] == ["SYD"]

    def test_code_filter_is_exact_not_partial(self, user):
        """Requirement 3 - Purpose: codes are identifiers, so a partial code matches nothing."""
        response = call("GET", "/airports?code=SY", user.token)
        assert response.json()["total"] == 0

    @pytest.mark.parametrize("wildcard", ["%", "_"])
    def test_sql_wildcards_matched_literally(self, user, wildcard):
        """Requirement 3 - Purpose: `%` and `_` typed by a client are literal characters, not SQL
        wildcards that would match everything."""
        response = call("GET", "/airports", user.token, params={"name": wildcard})
        assert response.json()["total"] == 0

    def test_airlines_operating_from_airport(self, user, data):
        """Requirement 3 - Purpose: the airlines operating from an airport are exactly the
        airlines with at least one route departing it."""
        expected = sorted(
            {int(r["airline_id"]) for r in data.routes if r["origin_airport_id"] == "1"}
        )
        response = call("GET", "/airports/1/airlines?per_page=100", user.token)
        assert response.status_code == 200
        body = response.json()
        assert_page(body, "airlines")
        assert [a["airline_id"] for a in body["airlines"]] == expected

    def test_airlines_of_unknown_airport(self, user):
        """Requirement 3 - Purpose: asking for the airlines of an unknown airport is a 404 rather
        than an empty list."""
        response = call("GET", "/airports/99999/airlines", user.token)
        assert response.status_code == 404
        assert response.json()["message"] == "Airport 99999 not found"

    def test_departing_routes_link(self, user, data):
        """Requirement 3 - Purpose: following an airport's `departing_routes` link returns the
        routes departing from it (navigation without knowing the database structure)."""
        airport = call("GET", "/airports/1", user.token).json()
        response = follow(airport["_links"]["departing_routes"] + "&per_page=100", user.token)
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == sum(r["origin_airport_id"] == "1" for r in data.routes)
        assert all(r["origin"]["iata_code"] == "SYD" for r in body["routes"])

    def test_arriving_routes_link(self, user, data):
        """Requirement 3 - Purpose: an airport's `arriving_routes` link lists routes into it."""
        airport = call("GET", "/airports/2", user.token).json()
        body = follow(airport["_links"]["arriving_routes"] + "&per_page=100", user.token).json()
        assert body["total"] == sum(r["destination_airport_id"] == "2" for r in data.routes)
        assert all(r["destination"]["iata_code"] == "MEL" for r in body["routes"])


class TestRequirement3Airlines:
    def test_list_airlines(self, user, data):
        """Requirement 3 - Purpose: all airlines are listed with the file's total."""
        response = call("GET", "/airlines?per_page=100", user.token)
        assert response.status_code == 200
        body = response.json()
        assert_page(body, "airlines")
        assert body["total"] == len(data.airlines)
        assert {a["iata_code"] for a in body["airlines"]} == {
            a["iata_code"] for a in data.airlines
        }

    def test_filter_airlines(self, user, data):
        """Requirement 3 - Purpose: airline name/country filters are partial and case-insensitive
        and `code` accepts IATA or ICAO."""
        body = call("GET", "/airlines?name=AIR&per_page=100", user.token).json()
        assert body["total"] == sum("air" in a["name"].lower() for a in data.airlines)

        body = call("GET", "/airlines?country=states&per_page=100", user.token).json()
        assert {a["country"] for a in body["airlines"]} == {"United States"}

        body = call("GET", "/airlines?code=qfa", user.token).json()
        assert [a["name"] for a in body["airlines"]] == ["Qantas"]

    def test_get_airline_and_follow_routes_link(self, user, data):
        """Requirement 3 - Purpose: an airline links to the routes it operates, and those are
        exactly its routes in the file."""
        response = call("GET", "/airlines/1", user.token)
        assert response.status_code == 200
        airline = response.json()
        assert airline["name"] == "Qantas"
        assert airline["_links"]["self"].endswith("/airlines/1")

        body = follow(airline["_links"]["routes"] + "&per_page=100", user.token).json()
        assert body["total"] == sum(r["airline_id"] == "1" for r in data.routes)
        assert all(r["airline"]["iata_code"] == "QF" for r in body["routes"])

    def test_get_unknown_airline(self, user):
        """Requirement 3 - Purpose: an unknown airline id returns 404 with a message."""
        response = call("GET", "/airlines/99999", user.token)
        assert response.status_code == 404
        assert response.json()["message"] == "Airline 99999 not found"


class TestRequirement3Routes:
    def test_list_routes_total(self, user, data):
        """Requirement 3 - Purpose: routes are paged and the total matches the file."""
        body = call("GET", "/routes?per_page=100&page=3", user.token).json()
        assert_page(body, "routes")
        assert body["total"] == len(data.routes)
        assert body["pages"] == -(-len(data.routes) // 100)
        assert len(body["routes"]) == len(data.routes) - 200

    def test_get_route_embeds_related_resources(self, user, data):
        """Requirement 3 - Purpose: a route embeds summaries of its airline and airports, each
        with an href that resolves to the full record."""
        expected = data.routes[0]
        response = call("GET", f"/routes/{expected['route_id']}", user.token)
        assert response.status_code == 200
        route = response.json()
        assert route["airline"]["iata_code"] == expected["airline_iata"]
        assert route["origin"]["iata_code"] == expected["origin_iata"]
        assert route["destination"]["iata_code"] == expected["destination_iata"]
        assert route["stops"] == int(expected["stops"])
        assert route["equipment"] == expected["equipment"]
        assert route["codeshare"] is (expected["codeshare"] == "Y")

        for part, id_key in (
            ("airline", "airline_id"),
            ("origin", "airport_id"),
            ("destination", "airport_id"),
        ):
            linked = follow(route[part]["href"], user.token)
            assert linked.status_code == 200
            assert linked.json()[id_key] == route[part][id_key]

    def test_get_unknown_route(self, user):
        """Requirement 3 - Purpose: an unknown route id returns 404 with a message."""
        response = call("GET", "/routes/99999", user.token)
        assert response.status_code == 404
        assert response.json()["message"] == "Route 99999 not found"

    @pytest.mark.parametrize(
        "query, matches",
        [
            pytest.param("airline=qf", lambda r: r["airline_iata"] == "QF", id="airline"),
            pytest.param("origin=syd", lambda r: r["origin_iata"] == "SYD", id="origin"),
            pytest.param(
                "destination=MEL", lambda r: r["destination_iata"] == "MEL", id="destination"
            ),
            pytest.param(
                "origin=SYD&destination=MEL&direct=true",
                lambda r: r["origin_iata"] == "SYD"
                and r["destination_iata"] == "MEL"
                and r["stops"] == "0",
                id="direct-between-airports",
            ),
            pytest.param("direct=false", lambda r: r["stops"] != "0", id="with-stops"),
            pytest.param("codeshare=true", lambda r: r["codeshare"] == "Y", id="codeshare"),
            pytest.param("codeshare=false", lambda r: r["codeshare"] == "", id="not-codeshare"),
            pytest.param("equipment=a320", lambda r: r["equipment"] == "A320", id="equipment"),
            pytest.param(
                "airline=QF&origin=SYD",
                lambda r: r["airline_iata"] == "QF" and r["origin_iata"] == "SYD",
                id="airline-and-origin",
            ),
        ],
    )
    def test_route_filters_match_csv(self, user, data, query, matches):
        """Requirement 3 - Purpose: each route filter (and combinations) returns exactly the
        routes in the file that satisfy it."""
        body = call("GET", f"/routes?{query}&per_page=100", user.token).json()
        expected = sorted(int(r["route_id"]) for r in data.routes if matches(r))
        assert expected, "the filter should match something in the sample data"
        assert body["total"] == len(expected)
        assert [r["route_id"] for r in body["routes"]] == expected[:100]

    def test_route_country_filters(self, user, data):
        """Requirement 3 - Purpose: origin/destination country filters use the airports'
        countries (partial match), e.g. Australia to Japan."""
        country = {a["airport_id"]: a["country"] for a in data.airports}
        expected = sorted(
            int(r["route_id"])
            for r in data.routes
            if country[r["origin_airport_id"]] == "Australia"
            and country[r["destination_airport_id"]] == "Japan"
        )
        body = call(
            "GET",
            "/routes?origin_country=australia&destination_country=JAP&per_page=100",
            user.token,
        ).json()
        assert [r["route_id"] for r in body["routes"]] == expected

    def test_unknown_code_returns_empty_page(self, user):
        """Requirement 3 - Purpose: a filter that matches nothing is an empty page, not a 404."""
        body = call("GET", "/routes?origin=XXX", user.token).json()
        assert body["total"] == 0 and body["routes"] == []

    def test_invalid_boolean_filter(self, user):
        """Requirement 3 - Purpose: a non-boolean value for `direct` is rejected with 400."""
        response = call("GET", "/routes?direct=maybe", user.token)
        assert_validation_error(response, "direct")


# ===========================================================================
# Requirement Set 4: My travel plans
# ===========================================================================
PLAN_KEYS = {"id", "name", "departure_date", "return_date", "route_ids", "_links"}


class TestRequirement4CreateAndRead:
    def test_create_plan(self, user):
        """Requirement 4 - Purpose: a user creates a plan; duplicate route ids are ignored, the
        response describes the plan and a Location header points to it."""
        payload = plan_payload(route_ids=[3, 1, 1])
        response = call("POST", "/travel-plans", user.token, json=payload)
        assert response.status_code == 201
        plan = response.json()
        assert set(plan) == PLAN_KEYS
        assert plan["name"] == payload["name"]
        assert plan["departure_date"] == payload["departure_date"]
        assert plan["return_date"] == payload["return_date"]
        assert plan["route_ids"] == [1, 3]
        assert response.headers["Location"].endswith(f"/travel-plans/{plan['id']}")

        fetched = follow(response.headers["Location"], user.token)
        assert fetched.status_code == 200
        assert fetched.json() == plan

    def test_create_plan_without_routes_and_same_day(self, user):
        """Requirement 4 - Purpose: routes are optional and a same-day trip (return date equal to
        departure date) is allowed."""
        day = days_from_today(10)
        plan = create_plan(user.token, departure_date=day, return_date=day)
        assert plan["route_ids"] == []

    def test_create_past_plan_allowed(self, user):
        """Requirement 4 - Purpose: past trips can be recorded (needed to tell future from past
        plans in Requirement 5)."""
        plan = create_plan(user.token, departure_date="2020-01-01", return_date="2020-01-10")
        assert plan["departure_date"] == "2020-01-01"

    def test_same_name_allowed(self, new_user):
        """Requirement 4 - Purpose: different accounts (and the same account) may reuse a plan
        name."""
        first, second = new_user(), new_user()
        name = unique("Holiday")
        create_plan(first.token, name=name)
        create_plan(first.token, name=name)
        create_plan(second.token, name=name)
        assert call("GET", "/travel-plans", first.token).json()["total"] == 2
        assert call("GET", "/travel-plans", second.token).json()["total"] == 1

    def test_return_before_departure_rejected(self, user):
        """Requirement 4 - Purpose: the return date must not be earlier than the departure date."""
        response = call(
            "POST",
            "/travel-plans",
            user.token,
            json=plan_payload(departure_date="2030-05-10", return_date="2030-05-01"),
        )
        assert response.status_code == 400
        assert response.json()["message"] == (
            "'return_date' must not be earlier than 'departure_date'"
        )

    @pytest.mark.parametrize("bad_date", ["10/05/2030", "2030-02-30", "tomorrow"])
    def test_malformed_date_rejected(self, user, bad_date):
        """Requirement 4 - Purpose: dates must be real YYYY-MM-DD dates."""
        response = call(
            "POST", "/travel-plans", user.token, json=plan_payload(departure_date=bad_date)
        )
        assert response.status_code == 400
        assert "'departure_date' must be a date in YYYY-MM-DD format" in response.json()[
            "message"
        ]

    def test_unknown_route_rejected(self, user):
        """Requirement 4 - Purpose: a plan cannot reference routes that do not exist."""
        response = call(
            "POST", "/travel-plans", user.token, json=plan_payload(route_ids=[1, 99999])
        )
        assert response.status_code == 400
        assert response.json()["message"] == "Unknown route id(s): [99999]"

    @pytest.mark.parametrize(
        "payload, field",
        [
            pytest.param({"departure_date": "2030-01-01", "return_date": "2030-01-02"}, "name",
                         id="missing-name"),
            pytest.param(plan_payload(name="x" * 101), "name", id="name-too-long"),
            pytest.param(plan_payload(route_ids=["one"]), "route_ids", id="non-integer-route"),
            pytest.param(plan_payload(owner_id=1), "", id="unknown-field"),
        ],
    )
    def test_invalid_payload_rejected(self, user, payload, field):
        """Requirement 4 - Purpose: the create payload is validated (required fields, lengths,
        types, and no unknown fields such as owner_id)."""
        response = call("POST", "/travel-plans", user.token, json=payload)
        assert_validation_error(response, field)

    def test_blank_name_rejected(self, user):
        """Requirement 4 - Purpose: a name made only of spaces is rejected."""
        response = call("POST", "/travel-plans", user.token, json=plan_payload(name="   "))
        assert response.status_code == 400
        assert response.json()["message"] == "'name' must not be blank"

    def test_list_own_plans_ordered_by_departure(self, user):
        """Requirement 4 - Purpose: a user lists only their own plans, ordered by departure date."""
        later = create_plan(user.token, departure_date=days_from_today(60),
                            return_date=days_from_today(61))
        earlier = create_plan(user.token, departure_date=days_from_today(5),
                              return_date=days_from_today(6))
        response = call("GET", "/travel-plans", user.token)
        assert response.status_code == 200
        body = response.json()
        assert_page(body, "travel_plans")
        assert [p["id"] for p in body["travel_plans"]] == [earlier["id"], later["id"]]

    def test_plan_routes_resource_gives_full_details(self, user, data):
        """Requirement 4 - Purpose: a consumer navigates from a plan to the full details of its
        routes, airlines and airports via `_links.routes`."""
        plan = create_plan(user.token, route_ids=[2, 1])
        response = follow(plan["_links"]["routes"], user.token)
        assert response.status_code == 200
        body = response.json()
        assert_page(body, "routes")
        assert [r["route_id"] for r in body["routes"]] == [1, 2]
        first = body["routes"][0]
        assert first["origin"]["iata_code"] == data.routes[0]["origin_iata"]
        assert follow(first["airline"]["href"], user.token).json()["name"] == "Qantas"

    def test_plans_require_login(self):
        """Requirement 4 - Purpose: travel plans are not available without a token."""
        assert call("GET", "/travel-plans").status_code == 401


class TestRequirement4Privacy:
    def test_other_users_plan_is_invisible(self, new_user):
        """Requirement 4 - Purpose: another User cannot read, change, delete or edit the routes of
        someone else's plan; every attempt is a 404, exactly like a missing plan."""
        owner, intruder = new_user(), new_user()
        plan = create_plan(owner.token, route_ids=[1])
        pid = plan["id"]
        for method, path, payload in (
            ("GET", f"/travel-plans/{pid}", None),
            ("PATCH", f"/travel-plans/{pid}", {"name": "hacked"}),
            ("DELETE", f"/travel-plans/{pid}", None),
            ("GET", f"/travel-plans/{pid}/routes", None),
            ("PUT", f"/travel-plans/{pid}/routes/2", None),
            ("DELETE", f"/travel-plans/{pid}/routes/1", None),
        ):
            response = call(method, path, intruder.token, json=payload)
            assert response.status_code == 404, (method, path)
            assert response.json()["message"] == f"Travel plan {pid} not found"

        assert call("GET", f"/travel-plans/{pid}", owner.token).json() == plan
        assert call("GET", "/travel-plans", intruder.token).json()["total"] == 0

    def test_admin_cannot_see_users_plans(self, admin_token, user):
        """Requirement 4 + 1 - Purpose: the admin can only see their own plans, not a User's."""
        plan = create_plan(user.token)
        response = call("GET", f"/travel-plans/{plan['id']}", admin_token)
        assert response.status_code == 404
        listed = call("GET", "/travel-plans?per_page=100", admin_token).json()["travel_plans"]
        assert plan["id"] not in [p["id"] for p in listed]

    def test_admin_manages_own_plans(self, admin_token, admin_plans):
        """Requirement 4 - Purpose: the admin can create and read their own travel plans."""
        plan = create_plan(admin_token)
        admin_plans.append(plan["id"])
        response = call("GET", f"/travel-plans/{plan['id']}", admin_token)
        assert response.status_code == 200
        assert response.json()["name"] == plan["name"]


class TestRequirement4Modify:
    def test_patch_name_only(self, user):
        """Requirement 4 - Purpose: PATCH changes only the fields that are sent."""
        plan = create_plan(user.token, route_ids=[1])
        response = call("PATCH", f"/travel-plans/{plan['id']}", user.token,
                        json={"name": "  Renamed  "})
        assert response.status_code == 200
        updated = response.json()
        assert updated["name"] == "Renamed"
        assert updated["departure_date"] == plan["departure_date"]
        assert updated["route_ids"] == [1]

    def test_patch_both_dates(self, user):
        """Requirement 4 - Purpose: both dates can be moved together."""
        plan = create_plan(user.token)
        response = call(
            "PATCH",
            f"/travel-plans/{plan['id']}",
            user.token,
            json={"departure_date": "2031-01-01", "return_date": "2031-01-09"},
        )
        assert response.status_code == 200
        assert response.json()["return_date"] == "2031-01-09"

    def test_patch_date_rule_uses_resulting_dates(self, user):
        """Requirement 4 - Purpose: moving only the departure date past the existing return date
        is rejected, because the rule is checked on the resulting plan."""
        plan = create_plan(user.token, departure_date="2030-03-01", return_date="2030-03-05")
        response = call("PATCH", f"/travel-plans/{plan['id']}", user.token,
                        json={"departure_date": "2030-03-06"})
        assert response.status_code == 400
        assert "must not be earlier" in response.json()["message"]
        assert call("GET", f"/travel-plans/{plan['id']}", user.token).json() == plan

    def test_patch_empty_body(self, user):
        """Requirement 4 - Purpose: an empty PATCH is rejected with a helpful message."""
        plan = create_plan(user.token)
        response = call("PATCH", f"/travel-plans/{plan['id']}", user.token, json={})
        assert response.status_code == 400
        assert response.json()["message"] == (
            "Provide at least one of: name, departure_date, return_date"
        )

    def test_patch_cannot_change_routes(self, user):
        """Requirement 4 - Purpose: routes are managed through the nested routes resource only,
        so `route_ids` is not accepted by PATCH."""
        plan = create_plan(user.token)
        response = call("PATCH", f"/travel-plans/{plan['id']}", user.token,
                        json={"route_ids": [1]})
        assert_validation_error(response, "")

    def test_delete_plan(self, user):
        """Requirement 4 - Purpose: deleting a plan removes it (204, then 404)."""
        plan = create_plan(user.token, route_ids=[1])
        response = call("DELETE", f"/travel-plans/{plan['id']}", user.token)
        assert response.status_code == 204
        assert call("GET", f"/travel-plans/{plan['id']}", user.token).status_code == 404
        assert call("DELETE", f"/travel-plans/{plan['id']}", user.token).status_code == 404

    def test_add_route_is_idempotent(self, user):
        """Requirement 4 - Purpose: PUT adds a route to a plan; repeating it changes nothing."""
        plan = create_plan(user.token, route_ids=[1])
        for _ in range(2):
            response = call("PUT", f"/travel-plans/{plan['id']}/routes/5", user.token)
            assert response.status_code == 200
            assert response.json()["route_ids"] == [1, 5]

    def test_add_unknown_route(self, user):
        """Requirement 4 - Purpose: adding a route that does not exist returns 404."""
        plan = create_plan(user.token)
        response = call("PUT", f"/travel-plans/{plan['id']}/routes/99999", user.token)
        assert response.status_code == 404
        assert response.json()["message"] == "Route 99999 not found"

    def test_remove_route(self, user):
        """Requirement 4 - Purpose: DELETE removes a route from the plan (the route itself still
        exists); removing it again is a 404."""
        plan = create_plan(user.token, route_ids=[1, 2])
        path = f"/travel-plans/{plan['id']}/routes/1"
        assert call("DELETE", path, user.token).status_code == 204
        assert call("GET", f"/travel-plans/{plan['id']}", user.token).json()["route_ids"] == [2]
        assert call("GET", "/routes/1", user.token).status_code == 200

        response = call("DELETE", path, user.token)
        assert response.status_code == 404
        assert response.json()["message"] == f"Route 1 is not in travel plan {plan['id']}"

    def test_unknown_plan(self, user):
        """Requirement 4 - Purpose: every plan endpoint returns 404 for a plan id that does not
        exist."""
        for method, path in (
            ("GET", "/travel-plans/99999999"),
            ("DELETE", "/travel-plans/99999999"),
            ("GET", "/travel-plans/99999999/routes"),
            ("PUT", "/travel-plans/99999999/routes/1"),
        ):
            response = call(method, path, user.token)
            assert response.status_code == 404, (method, path)
            assert response.json()["message"] == "Travel plan 99999999 not found"


# ===========================================================================
# Requirement Set 5: Exporting travel plans
# ===========================================================================
class TestRequirement5JsonExport:
    def test_json_export_is_downloadable_file(self, user):
        """Requirement 5 - Purpose: `format=json` returns a JSON attachment named after the user."""
        response = call("GET", "/export/travel-plans?format=json", user.token)
        assert response.status_code == 200
        assert response.headers["Content-Type"].startswith("application/json")
        disposition = response.headers["Content-Disposition"]
        assert disposition.startswith("attachment;")
        assert f"travel-plans_{user.username}_" in disposition
        assert response.headers["Cache-Control"] == "no-store"

    def test_json_export_empty_account(self, user):
        """Requirement 5 - Purpose: an account without plans still gets a valid, empty file."""
        body = call("GET", "/export/travel-plans?format=json", user.token).json()
        assert set(body) == {"owner", "exported_at", "total_plans", "travel_plans"}
        assert body["owner"] == user.username
        assert body["total_plans"] == 0
        assert body["travel_plans"] == []

    def test_json_export_ordered_and_self_contained(self, user, data):
        """Requirement 5 - Purpose: all plans are exported ordered by departure date, each with
        the full details of its routes and of their airlines and airports (no bare ids, no API
        links)."""
        late = create_plan(user.token, departure_date="2031-06-01", return_date="2031-06-05",
                           route_ids=[1])
        past = create_plan(user.token, departure_date="2021-06-01", return_date="2021-06-05")
        soon = create_plan(user.token, departure_date=days_from_today(3),
                           return_date=days_from_today(4), route_ids=[2, 1])

        response = call("GET", "/export/travel-plans?format=json", user.token)
        body = response.json()
        assert body["total_plans"] == 3
        assert [p["id"] for p in body["travel_plans"]] == [past["id"], soon["id"], late["id"]]

        plan = body["travel_plans"][1]
        assert set(plan) == {"id", "name", "departure_date", "return_date", "routes"}
        assert [r["route_id"] for r in plan["routes"]] == [1, 2]

        route = plan["routes"][0]
        csv_route = data.routes[0]
        csv_origin = next(a for a in data.airports if a["airport_id"] == csv_route["origin_airport_id"])
        csv_airline = next(a for a in data.airlines if a["airline_id"] == csv_route["airline_id"])
        assert route["airline"] == {
            "airline_id": int(csv_airline["airline_id"]),
            "name": csv_airline["name"],
            "iata_code": csv_airline["iata_code"],
            "icao_code": csv_airline["icao_code"],
            "country": csv_airline["country"],
        }
        assert route["origin"]["name"] == csv_origin["name"]
        assert route["origin"]["latitude"] == pytest.approx(float(csv_origin["latitude"]))
        assert route["origin"]["timezone"] == csv_origin["timezone"]
        assert "_links" not in response.text and "href" not in response.text

    def test_json_export_only_own_plans(self, new_user):
        """Requirement 5 - Purpose: an account exports only its own plans."""
        owner, other = new_user(), new_user()
        secret = create_plan(owner.token)
        body = call("GET", "/export/travel-plans?format=json", other.token).json()
        assert body["total_plans"] == 0
        assert secret["name"] not in str(body)

    def test_admin_exports_only_own_plans(self, admin_token, admin_plans, user):
        """Requirement 5 + 1 - Purpose: the admin's export contains the admin's plans only."""
        mine = create_plan(admin_token)
        admin_plans.append(mine["id"])
        theirs = create_plan(user.token)
        body = call("GET", "/export/travel-plans?format=json", admin_token).json()
        names = [p["name"] for p in body["travel_plans"]]
        assert body["owner"] == "admin"
        assert mine["name"] in names
        assert theirs["name"] not in names


class TestRequirement5PngExport:
    def test_png_timeline(self, user):
        """Requirement 5 - Purpose: `format=png` returns a PNG image shown inline in the browser
        when the account has future plans."""
        create_plan(user.token, departure_date=days_from_today(7),
                    return_date=days_from_today(9), route_ids=[1, 2, 3, 4])
        response = call("GET", "/export/travel-plans?format=png", user.token)
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.headers["Content-Disposition"].startswith("inline;")
        assert response.content.startswith(PNG_SIGNATURE)

    def test_png_with_more_than_five_future_plans(self, user):
        """Requirement 5 - Purpose: with more than five future plans the image is still produced
        (it shows the five nearest)."""
        for offset in range(1, 8):
            create_plan(user.token, departure_date=days_from_today(offset * 10),
                        return_date=days_from_today(offset * 10 + 2))
        response = call("GET", "/export/travel-plans?format=png", user.token)
        assert response.status_code == 200
        assert response.content.startswith(PNG_SIGNATURE)

    def test_png_without_plans(self, user):
        """Requirement 5 - Purpose: an account with no plans gets 404 and a JSON message instead
        of an image."""
        response = call("GET", "/export/travel-plans?format=png", user.token)
        assert response.status_code == 404
        assert response.headers["Content-Type"].startswith("application/json")
        assert "no future travel plans" in response.json()["message"]

    def test_png_ignores_past_and_today_plans(self, user):
        """Requirement 5 - Purpose: only plans departing strictly after today count as future,
        so past plans and a plan departing today give 404."""
        today = date.today().isoformat()
        create_plan(user.token, departure_date="2020-01-01", return_date="2020-01-05")
        create_plan(user.token, departure_date=today, return_date=days_from_today(2))
        response = call("GET", "/export/travel-plans?format=png", user.token)
        assert response.status_code == 404
        assert today in response.json()["message"]


class TestRequirement5FormatSelection:
    def test_format_is_required(self, user):
        """Requirement 5 - Purpose: the client must choose a format; there is no default."""
        response = call("GET", "/export/travel-plans", user.token)
        assert_validation_error(response, "format")

    @pytest.mark.parametrize("fmt", ["xml", "JSON", "pdf", ""])
    def test_unsupported_format(self, user, fmt):
        """Requirement 5 - Purpose: unsupported (or wrongly cased) formats are rejected with 400."""
        response = call("GET", f"/export/travel-plans?format={fmt}", user.token)
        assert_validation_error(response, "format")

    def test_export_requires_login(self):
        """Requirement 5 - Purpose: exporting requires authentication."""
        response = call("GET", "/export/travel-plans?format=json")
        assert response.status_code == 401
