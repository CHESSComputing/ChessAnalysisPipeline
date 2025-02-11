#!/usr/bin/env python
"""Processors used only by SAXSWAXS experiments."""

from CHAP import Processor

class PyfaiIntegrationProcessor(Processor):
    """Processor for performing pyFAI integrations."""
    def process(self, data, config=None,
                idx_slices=[{'start':0, 'end': -1, 'step': 1}]):
        import numpy as np
        import time

        # Get config for PyfaiIntegrationProcessor from data or config
        try:
            config = self.get_config(
                data, f'saxswaxs.models.{self.__class__.__name__}Config')
        except:
            self.logger.info(
                f'No valid {self.__class__.__name__} config in input '
                'pipeline data, using config parameter instead')
            try:
                from CHAP.saxswaxs.models import (
                    PyfaiIntegrationProcessorConfig)
                config = PyfaiIntegrationProcessorConfig(**config)
            except Exception as exc:
                self.logger.error(exc)
                raise RuntimeError(exc)

        # Organize input for integrations
        input_data = {d['name']: d['data'] for d in data}
        ais = {ai.name: ai.ai for ai in config.azimuthal_integrators}

        # Finalize idx slice for results
        idx = tuple(slice(idx_slice.get('start'),
                     idx_slice.get('end'),
                     idx_slice.get('step')) for idx_slice in idx_slices)
        # Perform integration(s), package results for ZarrResultsWriter
        results = []
        nframes = len(input_data[list(ais.keys())[0]])
        for i, integration in enumerate(config.integrations):
            t0 = time.time()
            self.logger.debug(f'Integrating {integration.name}...')
            result = integration.integrate(ais, input_data)
            tf = time.time()
            self.logger.debug(f'Integrated {integration.name} ({nframes/(tf-t0)} frames/sec)')
            results.extend(
                [
                    {
                        'path': f'{integration.name}/data/I',
                        'idx': idx,
                        'data': np.asarray([[r.intensity for r in result]]),
                    },
                ]
            )
        return results


if __name__ == '__main__':
    # Local modules
    from CHAP.processor import main

    main()
