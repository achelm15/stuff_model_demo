# Databricks notebook source
"""Operational helpers kept outside the narrated monitoring cells."""

import time

from databricks.sdk.errors import NotFound
from databricks.sdk.service.dataquality import (
    DataProfilingStatus,
    Monitor,
    Refresh,
    RefreshState,
)


def ensure_monitor(workspace, table_id, config, timeout_minutes=15):
    """Create the monitor once, or reuse the monitor already attached to the table."""
    try:
        monitor = workspace.data_quality.get_monitor(object_type="table", object_id=table_id)
        print("Reusing the existing Data Quality Inference profile.")
    except NotFound:
        monitor = workspace.data_quality.create_monitor(
            monitor=Monitor(
                object_type="table",
                object_id=table_id,
                data_profiling_config=config,
            )
        )
        print("Created the Data Quality Inference profile.")

    deadline = time.time() + timeout_minutes * 60
    while monitor.data_profiling_config.status == DataProfilingStatus.DATA_PROFILING_STATUS_PENDING:
        if time.time() >= deadline:
            raise TimeoutError("Data Quality profile creation timed out.")
        time.sleep(10)
        monitor = workspace.data_quality.get_monitor(object_type="table", object_id=table_id)

    if monitor.data_profiling_config.status != DataProfilingStatus.DATA_PROFILING_STATUS_ACTIVE:
        message = monitor.data_profiling_config.latest_monitor_failure_message or "No details returned."
        raise RuntimeError(f"Data Quality profile is not active: {message}")
    return monitor


def refresh_and_wait(workspace, table_id, timeout_minutes=45):
    """Start one refresh, reuse an overlapping refresh, and surface failures to the job."""
    running_states = {
        RefreshState.MONITOR_REFRESH_STATE_PENDING,
        RefreshState.MONITOR_REFRESH_STATE_RUNNING,
    }
    refresh = next(
        (
            item
            for item in workspace.data_quality.list_refresh(
                object_type="table", object_id=table_id
            )
            if item.state in running_states
        ),
        None,
    )
    if refresh is None:
        refresh = workspace.data_quality.create_refresh(
            object_type="table",
            object_id=table_id,
            refresh=Refresh(object_type="table", object_id=table_id),
        )

    deadline = time.time() + timeout_minutes * 60
    while refresh.state in running_states:
        if time.time() >= deadline:
            raise TimeoutError("Data Quality refresh timed out.")
        print(time.strftime("%H:%M:%S"), refresh.state)
        time.sleep(30)
        refresh = workspace.data_quality.get_refresh(
            object_type="table",
            object_id=table_id,
            refresh_id=refresh.refresh_id,
        )

    if refresh.state != RefreshState.MONITOR_REFRESH_STATE_SUCCESS:
        raise RuntimeError(f"Data Quality refresh ended in {refresh.state}.")
    return refresh
