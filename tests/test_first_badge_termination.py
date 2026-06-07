import server.app as server_app


class FakeMemoryReader:
    def __init__(self, badges, location, money=3000, coordinates=(5, 3), in_battle=False):
        self.badges = badges
        self.location = location
        self.money = money
        self.coordinates = coordinates
        self.in_battle = in_battle

    def read_badges(self):
        return list(self.badges)

    def read_location(self):
        return self.location

    def read_money(self):
        return self.money

    def read_coordinates(self):
        return self.coordinates

    def is_in_battle(self):
        return self.in_battle


class FakeMilestoneTracker:
    def __init__(self, completed=None):
        self.completed = set(completed or [])

    def is_completed(self, milestone):
        return milestone in self.completed


class FakeEnv:
    def __init__(self, reader, completed=None):
        self.memory_reader = reader
        self.milestone_tracker = FakeMilestoneTracker(completed)

    def get_location(self):
        return self.memory_reader.location

    def get_money(self):
        return self.memory_reader.money

    def get_coordinates(self):
        return self.memory_reader.coordinates


def set_env(monkeypatch, *, badges, location, money=3000, coordinates=(5, 3), completed=None, in_battle=False):
    reader = FakeMemoryReader(badges, location, money=money, coordinates=coordinates, in_battle=in_battle)
    monkeypatch.setattr(server_app, "env", FakeEnv(reader, completed=completed))
    monkeypatch.setattr(server_app, "first_badge_stable_frames", 0)
    return reader


def test_first_badge_rejects_corrupt_route_102_badge_read(monkeypatch):
    set_env(monkeypatch, badges=["Stone"], location="ROUTE 102", money=1459018779)

    status = server_app.get_first_badge_status()

    assert status["candidate_met"] is False
    assert status["condition_met"] is False
    assert status["stable_frames"] == 0
    assert "not_in_rustboro_gym_context" in status["validation_errors"]
    assert "money_out_of_range" in status["validation_errors"]


def test_first_badge_rejects_wrong_badge_outside_gym(monkeypatch):
    set_env(monkeypatch, badges=["Knuckle"], location="RUSTBORO CITY MART", money=1919)

    status = server_app.get_first_badge_status()

    assert status["candidate_met"] is False
    assert status["condition_met"] is False
    assert "stone_badge_not_in_memory" in status["validation_errors"]
    assert "not_in_rustboro_gym_context" in status["validation_errors"]


def test_first_badge_requires_stable_stone_badge_in_rustboro_gym(monkeypatch):
    set_env(
        monkeypatch,
        badges=["Stone"],
        location="RUSTBORO CITY GYM",
        money=6548,
        completed=["RUSTBORO_GYM_ENTERED"],
    )

    statuses = [server_app.get_first_badge_status() for _ in range(server_app.FIRST_BADGE_REQUIRED_STABLE_FRAMES)]

    assert [status["candidate_met"] for status in statuses] == [True] * server_app.FIRST_BADGE_REQUIRED_STABLE_FRAMES
    assert [status["stable_frames"] for status in statuses] == list(range(1, server_app.FIRST_BADGE_REQUIRED_STABLE_FRAMES + 1))
    assert statuses[-2]["condition_met"] is False
    assert statuses[-1]["condition_met"] is True


def test_first_badge_stability_resets_after_invalid_read(monkeypatch):
    reader = set_env(monkeypatch, badges=["Stone"], location="RUSTBORO CITY GYM", money=6548)
    assert server_app.get_first_badge_status()["stable_frames"] == 1
    assert server_app.get_first_badge_status()["stable_frames"] == 2

    reader.location = "PETALBURG WOODS"
    status = server_app.get_first_badge_status()

    assert status["candidate_met"] is False
    assert status["stable_frames"] == 0
    assert status["condition_met"] is False
