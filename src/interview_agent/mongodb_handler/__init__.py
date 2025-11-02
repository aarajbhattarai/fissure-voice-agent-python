from .connection_args import connection_args, connection_args_example

try:
    from interview_agent.mongodb_handler.mongodb_handler import (
        MongoDBHandler as Handler,
    )

    import_error = None
except Exception as e:
    Handler = None
    import_error = e


title = "MongoDB"
name = "mongodb"

__all__ = [
    "Handler",
    "connection_args",
    "connection_args_example",
    "import_error",
    "name",
    "title",
]
