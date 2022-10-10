"""Utilities for defining Python tests."""
import sys
import socket
from platform import system
from typing import TypedDict

PytestSkip = TypedDict("PytestSkip", {"condition": bool, "reason": str})


def has_ipv6() -> bool:
    """Check whether IPv6 is enabled on this host."""
    # connection error in macos, still need some fixes.
    if system() not in ("Linux", "Windows"):
        return False

    if socket.has_ipv6:
        try:
            with socket.socket(
                socket.AF_INET6, socket.SOCK_STREAM
            ) as server, socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as client:
                server.bind(("::1", 0))
                port = server.getsockname()[1]
                server.listen()

                client.connect(("::1", port))
                conn, _ = server.accept()

                client.sendall("abc".encode())
                msg = conn.recv(3).decode()
                # if the code can be executed to this point, the message should be
                # correct.
                assert msg == "abc"
            return True
        except OSError:
            pass
    return False


def skip_ipv6() -> PytestSkip:
    """PyTest skip mark for IPv6."""
    return {"condition": not has_ipv6(), "reason": "IPv6 is required to be enabled."}


def skip_spark() -> PytestSkip:
    """Check whether we should skip PySpark tests."""
    if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
        print("skip 0")
        return {"condition": True, "reason": "Current platform is not supported."}

    try:
        import pyspark          # noqa
        SPARK_INSTALLED = True
    except ImportError:
        SPARK_INSTALLED = False

    return {"condition": not SPARK_INSTALLED, "reason": "Spark is not installed"}


def add_path(path: str):
    """Add to system path."""
    if path not in sys.path:
        sys.path.append(path)
