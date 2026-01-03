"""
Job Tracker Module

Tracks workflow execution jobs with persistent history, status monitoring,
and output management.
"""

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a workflow execution job."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobOutput:
    """Represents an output from a job."""
    node_id: str
    output_type: str
    data: Any
    filename: Optional[str] = None
    preview_url: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "output_type": self.output_type,
            "data": self.data if not isinstance(self.data, bytes) else "<binary>",
            "filename": self.filename,
            "preview_url": self.preview_url
        }


@dataclass
class Job:
    """Represents a workflow execution job."""
    job_id: str
    prompt_id: str
    workflow_name: str
    status: JobStatus
    created_at: datetime
    connection_name: str = "default"
    inputs: dict = field(default_factory=dict)
    outputs: list[JobOutput] = field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    current_node: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None

    @property
    def is_active(self) -> bool:
        """Check if job is still active."""
        return self.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "prompt_id": self.prompt_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "connection_name": self.connection_name,
            "inputs": self.inputs,
            "outputs": [o.to_dict() for o in self.outputs],
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "progress": self.progress,
            "current_node": self.current_node,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create a Job from a dictionary."""
        outputs = [
            JobOutput(
                node_id=o["node_id"],
                output_type=o["output_type"],
                data=o.get("data"),
                filename=o.get("filename"),
                preview_url=o.get("preview_url")
            )
            for o in data.get("outputs", [])
        ]

        return cls(
            job_id=data["job_id"],
            prompt_id=data["prompt_id"],
            workflow_name=data["workflow_name"],
            status=JobStatus(data["status"]),
            connection_name=data.get("connection_name", "default"),
            inputs=data.get("inputs", {}),
            outputs=outputs,
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=data.get("progress", 0.0),
            current_node=data.get("current_node"),
            metadata=data.get("metadata", {})
        )


class JobTracker:
    """
    Tracks workflow execution jobs with persistent history.

    Features:
    - Job creation and status tracking
    - Persistent history storage
    - Query jobs by status, workflow, or time range
    - Statistics and analytics
    """

    def __init__(
        self,
        history_file: Optional[Path] = None,
        max_history: int = 1000,
        auto_save: bool = True
    ):
        self.history_file = history_file
        self.max_history = max_history
        self.auto_save = auto_save

        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()

        if history_file and history_file.exists():
            self._load_history()

    def _load_history(self) -> None:
        """Load job history from file."""
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
                for job_data in data.get("jobs", []):
                    try:
                        job = Job.from_dict(job_data)
                        self._jobs[job.job_id] = job
                    except Exception as e:
                        logger.warning(f"Failed to load job: {e}")
            logger.info(f"Loaded {len(self._jobs)} jobs from history")
        except Exception as e:
            logger.error(f"Failed to load job history: {e}")

    def _save_history(self) -> None:
        """Save job history to file."""
        if not self.history_file:
            return

        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            # Get most recent jobs up to max_history
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )[:self.max_history]

            data = {
                "version": "2.0",
                "saved_at": datetime.now().isoformat(),
                "jobs": [j.to_dict() for j in jobs]
            }

            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(jobs)} jobs to history")
        except Exception as e:
            logger.error(f"Failed to save job history: {e}")

    def create_job(
        self,
        job_id: str,
        prompt_id: str,
        workflow_name: str,
        connection_name: str = "default",
        inputs: Optional[dict] = None,
        metadata: Optional[dict] = None
    ) -> Job:
        """
        Create a new job.

        Args:
            job_id: Unique job identifier
            prompt_id: ComfyUI prompt ID
            workflow_name: Name of the workflow
            connection_name: Name of the ComfyUI connection
            inputs: Input parameters used
            metadata: Additional metadata

        Returns:
            Created Job object
        """
        with self._lock:
            job = Job(
                job_id=job_id,
                prompt_id=prompt_id,
                workflow_name=workflow_name,
                status=JobStatus.PENDING,
                connection_name=connection_name,
                inputs=inputs or {},
                metadata=metadata or {},
                created_at=datetime.now()
            )
            self._jobs[job_id] = job

            if self.auto_save:
                self._save_history()

            logger.info(f"Created job: {job_id} for workflow {workflow_name}")
            return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_by_prompt(self, prompt_id: str) -> Optional[Job]:
        """Get a job by its ComfyUI prompt ID."""
        with self._lock:
            for job in self._jobs.values():
                if job.prompt_id == prompt_id:
                    return job
            return None

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        progress: Optional[float] = None,
        current_node: Optional[str] = None
    ) -> Optional[Job]:
        """
        Update job status.

        Args:
            job_id: Job ID to update
            status: New status
            error: Error message if failed
            progress: Progress percentage (0-1)
            current_node: Currently executing node

        Returns:
            Updated Job or None if not found
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            old_status = job.status
            job.status = status

            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now()

            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job.completed_at = datetime.now()
                if status == JobStatus.COMPLETED:
                    job.progress = 1.0

            if error:
                job.error = error

            if progress is not None:
                job.progress = progress

            if current_node is not None:
                job.current_node = current_node

            if self.auto_save and old_status != status:
                self._save_history()

            logger.debug(f"Updated job {job_id}: {old_status.value} -> {status.value}")
            return job

    def add_output(
        self,
        job_id: str,
        node_id: str,
        output_type: str,
        data: Any,
        filename: Optional[str] = None,
        preview_url: Optional[str] = None
    ) -> Optional[Job]:
        """Add an output to a job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None

            output = JobOutput(
                node_id=node_id,
                output_type=output_type,
                data=data,
                filename=filename,
                preview_url=preview_url
            )
            job.outputs.append(output)

            if self.auto_save:
                self._save_history()

            return job

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from history."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                if self.auto_save:
                    self._save_history()
                return True
            return False

    def clear_history(self, keep_active: bool = True) -> int:
        """
        Clear job history.

        Args:
            keep_active: If True, keep active jobs

        Returns:
            Number of jobs removed
        """
        with self._lock:
            if keep_active:
                to_remove = [
                    job_id for job_id, job in self._jobs.items()
                    if not job.is_active
                ]
            else:
                to_remove = list(self._jobs.keys())

            for job_id in to_remove:
                del self._jobs[job_id]

            if self.auto_save:
                self._save_history()

            return len(to_remove)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        workflow_name: Optional[str] = None,
        connection_name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            workflow_name: Filter by workflow name
            connection_name: Filter by connection
            since: Only jobs created after this time
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of matching jobs
        """
        with self._lock:
            jobs = list(self._jobs.values())

            # Apply filters
            if status:
                jobs = [j for j in jobs if j.status == status]

            if workflow_name:
                jobs = [j for j in jobs if j.workflow_name == workflow_name]

            if connection_name:
                jobs = [j for j in jobs if j.connection_name == connection_name]

            if since:
                jobs = [j for j in jobs if j.created_at >= since]

            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)

            # Apply pagination
            return jobs[offset:offset + limit]

    def get_active_jobs(self) -> list[Job]:
        """Get all active (pending, queued, running) jobs."""
        return self.list_jobs(status=None, limit=1000)

    def get_statistics(self) -> dict:
        """
        Get job statistics.

        Returns:
            Dict with various statistics
        """
        with self._lock:
            jobs = list(self._jobs.values())

            # Count by status
            status_counts = {}
            for status in JobStatus:
                status_counts[status.value] = sum(1 for j in jobs if j.status == status)

            # Count by workflow
            workflow_counts = {}
            for job in jobs:
                workflow_counts[job.workflow_name] = workflow_counts.get(job.workflow_name, 0) + 1

            # Calculate average duration for completed jobs
            completed = [j for j in jobs if j.status == JobStatus.COMPLETED and j.duration]
            avg_duration = (
                sum(j.duration for j in completed) / len(completed)
                if completed else 0
            )

            # Success rate
            finished = [j for j in jobs if j.status in (JobStatus.COMPLETED, JobStatus.FAILED)]
            success_rate = (
                sum(1 for j in finished if j.status == JobStatus.COMPLETED) / len(finished)
                if finished else 0
            )

            return {
                "total_jobs": len(jobs),
                "status_counts": status_counts,
                "workflow_counts": workflow_counts,
                "average_duration_seconds": avg_duration,
                "success_rate": success_rate,
                "active_jobs": sum(1 for j in jobs if j.is_active)
            }

    def get_recent_failures(self, limit: int = 10) -> list[Job]:
        """Get recent failed jobs."""
        return self.list_jobs(status=JobStatus.FAILED, limit=limit)

    def get_recent_successes(self, limit: int = 10) -> list[Job]:
        """Get recent successful jobs."""
        return self.list_jobs(status=JobStatus.COMPLETED, limit=limit)


# Global job tracker instance
_global_tracker: Optional[JobTracker] = None


def get_job_tracker(history_file: Optional[Path] = None) -> JobTracker:
    """Get or create the global job tracker."""
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = JobTracker(history_file=history_file)

    return _global_tracker


def reset_job_tracker() -> None:
    """Reset the global job tracker."""
    global _global_tracker
    _global_tracker = None
