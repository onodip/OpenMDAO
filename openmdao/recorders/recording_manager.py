"""
RecordingManager class definition.
"""
from six import iteritems
import time

import numpy as np

from openmdao.utils.mpi import MPI



class RecordingManager(object):
    """
    Object that routes function calls to all attached recorders.

    Attributes
    ----------
    _recorders : list of <BaseRecorder>
        All of the recorders attached to the current object.
    rank : int
        Rank of the iteration coordinate.
    """

    def __init__(self):
        """
        init.
        """

        self._vars_to_record = {
            'desvarnames': set(),
            'responsenames': set(),
            'objectivenames': set(),
            'constraintnames': set(),
            }

        self._recorders = []
        self._has_serial_recorders = False

        if MPI:
            self.rank = MPI.COMM_WORLD.rank
        else:
            self.rank = 0

    def __getitem__(self, index):
        """
        Get a particular recorder in the manager.
        """
        return self._recorders[index]

    def __iter__(self):
        """
        Iterate.

        Returns
        -------
        iter: <BaseRecorder>
            a recorder from _recorders
        """
        return iter(self._recorders)

    def _gather_vars(self, root, local_vars):
        """Gathers and returns only variables listed in
        `local_vars` from the `root` System.
        """

        # if trace:
        #     debug("gathering vars for recording in %s" % root.pathname)
        all_vars = root.comm.gather(local_vars, root=0)
        # if trace:
        #     debug("DONE gathering rec vars for %s" % root.pathname)

        if root.comm.rank == 0:
            dct = all_vars[-1]
            for d in all_vars[:-1]:
                dct.update(d)
            return dct

    def append(self, recorder):
        """
        Add a recorder for recording.

        Parameters
        ----------
        recorder : <BaseRecorder>
           Recorder instance to be added to the manager.
        """
        self._recorders.append(recorder)

    def startup(self, object_requesting_recording):
        """
        Run startup on each recorder in the manager.
        """
        #qqq pathname = object_requesting_recording.pathname
        # have to check to see if a given system is in the `_subsystems_myproc` list to see if it's active

        # Only the root will have _subsystems_myproc
        # Will only add parallel code for Drivers. Use the old method for System and Solver
        from openmdao.core.driver import Driver
        if not isinstance(object_requesting_recording,Driver):
            for recorder in self._recorders:
                recorder.startup(object_requesting_recording)
            return

        # object_requesting_recording._problem exists, which has these attributes:
        #   comm
        #   driver
        #   model ( aka root )

        # object_requesting_recording._problem.model has
        #      _subsystems_myproc
        #      _subsystems_allprocs
        model = object_requesting_recording._problem.model
        # if MPI:
        if MPI and model.is_active():
            rrank = object_requesting_recording._problem.comm.rank # root ( aka model ) rank. So this only works for
                # Driver recording
            # rowned = object_requesting_recording._problem._owning_ranks
            # in clippy, rowned is supposed to map variables to the ranks that own them.
            # qqq = model._var_allprocs_abs2idx['output']
            # out_var_idx = model._var_allprocs_abs2idx['output'][output_name]
            # root = np.min(np.nonzero(model._var_sizes[:, out_var_idx])[0][0])

            # Compute owning ranks
            rowned = {}
            for varname, out_var_idx in iteritems(model._var_allprocs_abs2idx['output']):
                rowned[varname] = np.min(np.nonzero(model._var_sizes['output'][:, out_var_idx])[0][0])

        self._record_desvars = self._record_responses = False
        self._record_objectives = self._record_constraints = False


        for recorder in self._recorders:
            recorder.startup(object_requesting_recording)

            if not recorder._parallel:
                self._has_serial_recorders = True

            # desvarnames = recorder._filtered[pathname]['desvars']
            # responsenames = recorder._filtered[pathname]['responses']
            # objectivenames = recorder._filtered[pathname]['objectives']
            # constraintnames = recorder._filtered[pathname]['constraints']

            desvarnames = recorder._filtered_driver['des']
            responsenames = recorder._filtered_driver['res']
            objectivenames = recorder._filtered_driver['obj']
            constraintnames = recorder._filtered_driver['con']

            if desvarnames:
                self._record_desvars = True
            if responsenames:
                self._record_responses = True
            if objectivenames:
                self._record_objectives = True
            if constraintnames:
                self._record_constraints = True



            # return # qqq

            # now localize the lists to only
            # include local vars.  We need to do this after determining
            # if any mpi procs need to record each of params, unknowns,
            # and resids.  If none of them do, we can skip the mpi gather
            # for that group of vars.
            if MPI:
                desvarnames = [n for n in desvarnames if rrank==rowned[n]]
                responsenames = [n for n in responsenames if rrank==rowned[n]]
                objectivenames = [n for n in objectivenames if rrank==rowned[n]]
                constraintnames = [n for n in constraintnames if rrank==rowned[n]]

                # reduce the filter set for any parallel recorders to only
                # those variables that are owned by that rank
                if recorder._parallel:
                    recorder._filtered_driver['des'] = desvarnames
                    recorder._filtered_driver['res'] = responsenames
                    recorder._filtered_driver['obj'] = objectivenames
                    recorder._filtered_driver['con'] = constraintnames

            self._vars_to_record['desvarnames'].update(desvarnames)
            self._vars_to_record['responsenames'].update(responsenames)
            self._vars_to_record['objectivenames'].update(objectivenames)
            self._vars_to_record['constraintnames'].update(constraintnames)




    def close(self):
        """
        Close all recorders in the manager.
        """
        for recorder in self._recorders:
            recorder.close()

    def record_iteration(self, object_requesting_recording, metadata, **kwargs):
        """
        Call record_iteration on all recorders.

        Parameters
        ----------
        object_requesting_recording : <object>
            The object that needs an iteration of itself recorded.
        metadata : dict
            Metadata for iteration coordinate
        **kwargs :
            Keyword args needed for different versions of record_iteration
        """
        if not self._recorders:
            return

        if metadata is not None:
            metadata['timestamp'] = time.time()

        from openmdao.core.driver import Driver

        if isinstance(object_requesting_recording, Driver):
            root = model = object_requesting_recording._problem.model

            desvars = object_requesting_recording.get_design_var_values()
            responses = object_requesting_recording.get_response_values()
            objectives = object_requesting_recording.get_objective_values()
            constraints = object_requesting_recording.get_constraint_values()

            if MPI:
                desvarnames = self._vars_to_record['desvarnames']
                responsenames = self._vars_to_record['responsenames']
                objectivenames = self._vars_to_record['objectivenames']
                constraintnames = self._vars_to_record['constraintnames']

                # get names and values of all locally owned variables
                if desvars:
                    desvars = {d: desvars[d] for d in desvarnames}
                if responses:
                    responses = {r: responses[r] for r in responsenames}
                if objectives:
                    objectives = {o: objectives[o] for o in objectivenames}
                if constraints:
                    constraints = {c: constraints[c] for c in constraintnames}

                if self._has_serial_recorders:
                    desvars = self._gather_vars(root, desvars) if self._record_desvars else {}
                    responses = self._gather_vars(root, responses) if self._record_responses else {}
                    objectives = self._gather_vars(root, objectives) if self._record_objectives else {}
                    constraints = self._gather_vars(root, constraints) if self._record_constraints else {}

        # If the recorder does not support parallel recording
        # we need to make sure we only record on rank 0.
        # for params, unknowns, resids, meta in cases:
        #     if params is None: # dummy cases have None in place of params, etc.
        #         continue
        for recorder in self._recorders:
            if recorder._parallel or MPI is None or self.rank == 0:
                # recorder.record_iteration(params, unknowns, resids, meta)
                if isinstance(object_requesting_recording, Driver):
                    recorder.record_iteration_driver_passing_vars(object_requesting_recording, desvars, responses, objectives, constraints, metadata)
                else:
                    recorder.record_iteration(object_requesting_recording, metadata, **kwargs)

        # Old serial way
        # for recorder in self._recorders:
        #     if recorder._parallel or MPI is None or self.rank == 0:
        #         recorder.record_iteration(object_requesting_recording, metadata, **kwargs)

    def record_metadata(self, object_requesting_recording):
        """
        Call record_metadata for all recorders.

        Parameters
        ----------
        object_requesting_recording : <object>
            The object that needs its metadata recorded.

        """
        for recorder in self._recorders:
            # If the recorder does not support parallel recording
            # we need to make sure we only record on rank 0.
            if recorder._parallel or self.rank == 0:
                recorder.record_metadata(object_requesting_recording)
