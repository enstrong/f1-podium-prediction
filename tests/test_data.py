from core import data


def _res(pos, did, grid, text=None, pts=0):
    return {"position": str(pos), "positionText": text or str(pos), "points": str(pts), "grid": str(grid),
            "status": "Finished", "Driver": {"driverId": did, "code": did[:3].upper(), "givenName": "A", "familyName": did},
            "Constructor": {"constructorId": "t", "name": "T"}}


def _race(rnd, key, rows):
    return {"season": "2025", "round": str(rnd), "raceName": f"GP {rnd}", "date": "2025-03-01",
            "Circuit": {"circuitId": f"c{rnd}", "Location": {"locality": "x", "country": "y"}}, key: rows}


def test_pagination_merge_and_build(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "RAW_DIR", tmp_path)
    monkeypatch.setattr(data, "PAGE", 4)
    pages = {  # race 2 split across pages
        ("results", 0): {"total": "6", "RaceTable": {"Races": [
            _race(1, "Results", [_res(1, "max", 1, pts=25), _res(2, "lando", 0, pts=18), _res(3, "oscar", 2, "R")]),
            _race(2, "Results", [_res(1, "lando", 2)])]}},
        ("results", 4): {"total": "6", "RaceTable": {"Races": [
            _race(2, "Results", [_res(2, "max", 1), _res(3, "oscar", 3)])]}},
        ("qualifying", 0): {"total": "3", "RaceTable": {"Races": [
            _race(1, "QualifyingResults", [{"position": "1", "Driver": {"driverId": "max"}},
                                           {"position": "2", "Driver": {"driverId": "oscar"}},
                                           {"position": "3", "Driver": {"driverId": "lando"}}])]}},
        ("sprint", 0): {"total": "1", "RaceTable": {"Races": [
            _race(2, "SprintResults", [{"position": "1", "points": "8", "Driver": {"driverId": "max"}}])]}},
    }
    monkeypatch.setattr(data, "_get", lambda url, p: pages[(url.split("/")[-1][:-5], p["offset"])])
    df = data.build_season(2025)
    data.validate(df)
    assert len(df) == 6
    r1 = df[df["round"] == 1].set_index("driver_id")
    assert r1.loc["lando", "pitlane_start"] and r1.loc["lando", "grid"] == 3
    assert r1.loc["oscar", "dnf"] and not r1.loc["max", "dnf"]
    assert r1.loc["lando", "quali"] == 3
    assert not r1["is_sprint_weekend"].any()
    r2 = df[df["round"] == 2].set_index("driver_id")
    assert r2["is_sprint_weekend"].all() and r2.loc["max", "sprint_points"] == 8
    assert (tmp_path / "2025_results.json").exists()
