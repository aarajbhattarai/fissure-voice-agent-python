from collections import OrderedDict

connection_args = OrderedDict(
    username={
        "type": "str",
        "description": "The username used to authenticate with the MongoDB server.",
        "required": True,
        "label": "User",
    },
    password={
        "type": "str",
        "description": "The password to authenticate the user with the MongoDB server.",
        "required": True,
        "label": "Password",
        "secret": True,
    },
    database={
        "type": "str",
        "description": "The database name to use when connecting with the MongoDB server.",
        "required": False,
        "label": "Database",
    },
    host={
        "type": "str",
        "description": "The host name or IP address of the MongoDB server. NOTE: use '127.0.0.1' instead of 'localhost' to connect to local server.",
        "required": True,
        "label": "Host",
    },
    port={
        "type": "int",
        "description": "The TCP/IP port of the MongoDB server. Must be an integer.",
        "required": True,
        "label": "Port",
    },
)

connection_args_example = OrderedDict(
    host="127.0.0.1",
    port=27017,
    username="mongo",
    password="password",
    database="database",
)
