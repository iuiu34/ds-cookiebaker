"""Utils."""
import datetime as dt
import os
from collections import namedtuple
from typing import NamedTuple

from .get_variables_types import get_variable_types  # noqa
from .release_model import release_model  # noqa


def get_output_path(base_output_path: str,
                    cache_timestamp: str = None
                    ) -> str:
    """Get output_path with execution time as id."""
    if cache_timestamp is None:
        cache_timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    out = os.path.join(base_output_path, str(cache_timestamp))
    return out


def dict_to_namedtuple(name, output=None):
    """Map dict to namedtuple."""
    if output is None:
        output = name
        name = 'Output'
    example_output = namedtuple(name,
                                list(output.keys())
                                )
    return example_output(*list(output.values()))


def get_params(dev: bool,
               output_path: str,
               start_date: str,
               end_date_days: int,
               end_valid_date_days: int,
               machine_type: str,
               memory_limit: str,
               n_iter: int,
               n_jobs_cv: int,
               n_splits: int,
               free_trial_days: int = 60) -> NamedTuple(
    'Outputs',
    [
        ('start_date', str),
        ('end_date', str),
        ('end_valid_date', str),
        ('n_iter', int),
        ('n_jobs_cv', int),
        ('n_splits', int),
        ('machine_type', str),
        ('memory_limit', str),
        ('output_path', str)
    ]):  # noqa
    """Get params for pipeline."""
    # output_path = output_path
    n_iter, n_jobs_cv = map(
        lambda x: None if x == -1 else x,
        [n_iter, n_jobs_cv]
    )

    if dev:
        end_date_days = 1
        end_valid_date_days = 1
        n_iter = 2  # hp searches
        n_jobs_cv = 2  # parallel threats in hp
        n_splits = 2  # cv folders
        memory_limit = '1G'
        machine_type = 'e2-standard-4'
    else:
        if memory_limit is None:
            memory_limit = '16G'
        if machine_type is None:
            machine_type = 'e2-highmem-8'
        if n_jobs_cv is None:
            n_jobs_cv = n_iter

    if start_date is None:
        start_date_days = free_trial_days + end_date_days + end_valid_date_days
        start_date_ = dt.date.today()
        start_date_ = start_date_ - dt.timedelta(days=start_date_days)
    else:
        start_date_ = dt.date.fromisoformat(start_date)

    end_date_ = start_date_ + dt.timedelta(days=end_date_days)
    end_valid_date_ = end_date_ + dt.timedelta(days=end_valid_date_days)

    start_date = start_date_.isoformat()
    end_date = end_date_.isoformat()
    end_valid_date = end_valid_date_.isoformat()
    output = dict(start_date=start_date,
                  end_date=end_date,
                  end_valid_date=end_valid_date,
                  n_iter=n_iter,
                  n_jobs_cv=n_jobs_cv,
                  n_splits=n_splits,
                  memory_limit=memory_limit,
                  machine_type=machine_type,
                  output_path=output_path)

    return dict_to_namedtuple(output)
