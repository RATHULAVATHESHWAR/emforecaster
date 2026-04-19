# Dummy Neptune (disable completely)
class DummyNeptune:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

neptune = DummyNeptune()


def log_pydantic(*args, **kwargs):
    print("Logging config (disabled)")


def epoch_logger(*args, **kwargs):
    print("Epoch log:", args)


def format_time_dynamic(seconds):
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    return f"{int(hrs)}h {int(mins)}m {int(secs)}s"


def global_to_yaml(*args, **kwargs):
    print("Saving config (disabled)")
