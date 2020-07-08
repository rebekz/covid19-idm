try:
    from fabric.cli import app
except ImportError:
    raise ImportError("fabric module is not installed. Please install fabric wheel located in the lib/ folder")

if __name__ == "__main__":
    app()
