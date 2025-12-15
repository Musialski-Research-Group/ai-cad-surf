import subprocess
import multiprocessing


def mp_worker(call):
    """Starts a new thread with a system call. Used for thread pooling."""
    call = call.split(' ')
    verbose = call[-1] == '--verbose'
    if verbose:
        call = call[:-1]
        subprocess.run(call)
    else:
        subprocess.run(
            call,
            stdout=subprocess.DEVNULL,
            # stderr=subprocess.DEVNULL
            )


def start_process_pool(worker_function, parameters, num_processes, timeout=None):
    """
    Executes a worker function in parallel using a process pool or a single loop
    based on the number of processes.
    """
    if len(parameters) <= 0:
        return None
    
    if num_processes <= 1:
        print('Running loop for {} with {} calls on {} workers'.format(
            str(worker_function), len(parameters), num_processes))
        results = []
        for c in parameters:
            results.append(worker_function(*c))
        return results
    print('Running loop for {} with {} calls on {} subprocess workers'.format(
        str(worker_function), len(parameters), num_processes))
    with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
        results = pool.starmap(worker_function, parameters)
        return results
