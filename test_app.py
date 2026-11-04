from fastapi.testclient import TestClient
import app as app_module


client = TestClient(app_module.app)


# Test 1: Home API
def test_home_api():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


# Test 2: Health API
def test_health_api():

    response = client.get("/health")

    assert response.status_code == 200
    assert "model_loaded" in response.json()


# Test 3: Successful prediction
def test_successful_prediction(monkeypatch):

    class FakeModel:

        def predict(self, data):
            return [1]

    monkeypatch.setattr(
        app_module,
        "model",
        FakeModel()
    )

    student_data = {
        "CGPA": 8.5,
        "Ten_Percentage": 85,
        "Twelve_Percentage": 80,
        "Internships": 2,
        "Projects": 3,
        "Aptitude_Score": 85,
        "Communication_Skills": 8
    }

    response = client.post("/predict", json=student_data)

    assert response.status_code == 200
    assert response.json()["placement_prediction"] == "Placed"


# Test 4: Invalid CGPA
def test_invalid_cgpa():

    student_data = {
        "CGPA": 15,
        "Ten_Percentage": 85,
        "Twelve_Percentage": 80,
        "Internships": 2,
        "Projects": 3,
        "Aptitude_Score": 85,
        "Communication_Skills": 8
    }

    response = client.post("/predict", json=student_data)

    assert response.status_code == 422


# Test 5: Missing input
def test_missing_input():

    response = client.post("/predict", json={})

    assert response.status_code == 422