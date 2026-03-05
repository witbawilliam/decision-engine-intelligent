from fastapi import FastAPI
from workers.tasks_training import run_mock_analysis

app = FastAPI(title="AI Data Analyst System")

@app.post("/analyze/{job_id}")
async def start_analysis(job_id: str):
    # .delay() pushes the task to Redis immediately
    task = run_mock_analysis.delay(job_id)
    return {"status": "Job Sent to Celery", "task_id": task.id}