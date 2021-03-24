from flask import request, jsonify
from . import main


def error_handler(code, message):
    response = jsonify({"error": message})
    response.status_code = code
    return response


@main.app_errorhandler(403)
def forbidden(e):
    return error_handler(403, "forbidden")


@main.app_errorhandler(400)
def page_not_found(e):
    if hasattr(e, "description") and e.description:
        message = e.description
    else:
        message = "bad request"
    return error_handler(400, message)


@main.app_errorhandler(404)
def page_not_found(e):
    if hasattr(e, "description") and e.description:
        message = e.description
    else:
        message = "not found"
    return error_handler(404, message)


@main.app_errorhandler(500)
def internal_server_error(e):
    return error_handler(500, "internal server error")