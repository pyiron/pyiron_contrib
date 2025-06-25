from multiprocessing import Pool, cpu_count

def parallelise(func, *args_list, **kwargs_list):
    """
    Executes the given function in parallel by applying it to multiple sets of arguments.

    Args:
        func: The function to be executed in parallel.
        max_workers: maximum number of processes to run in parallel
        *args_list: Variable-length argument list containing iterables, each representing a list of arguments
                    to be passed to the function. The function will be called for each set of arguments in parallel.
        **kwargs_list: Variable-length keyword argument list containing keyword arguments to be passed to the function.

    Returns:
        List: A list containing the results of executing the function for each set of arguments.

    Example:
        def func(filename, filepath, suffix="asdf"):
            # Your function logic here
            # ...
            return result

        filenames = ["file1.txt", "file2.txt", "file3.txt"]
        filepaths = ["/path/to/file1", "/path/to/file2", "/path/to/file3"]
        suffixes = ["suffix1", "suffix2", "suffix3"]

        result = parallelise(func, filenames, filepaths, suffixes)
    """
    if len(args_list[0]) == 0:
        return None
    else:
        # if max_workers:
        #     num_processors = min(len(args_list[0]), max_workers)
        # else:
        num_processors = min(len(args_list[0]), cpu_count())
        print(f"# Processes: {len(args_list[0])}\nProcessors available: {cpu_count()}\nCPUs used: {num_processors}")

        # Number of processes to run in parallel
        pool = Pool(processes=num_processors)

        if kwargs_list:
            results = pool.starmap(func, zip(*args_list), **kwargs_list)            
        else:
            results = pool.starmap(func, zip(*args_list))
        # # Map the function call to each set of arguments using multiprocessing
        # results = pool.starmap(func, zip(*args_list), **kwargs_list)

        pool.close()
        pool.join()

    return results
