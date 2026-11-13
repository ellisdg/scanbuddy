import os
import sys
import pdb
import time
import math
import json
import shutil
import psutil
import logging
import datetime
import threading
import numpy as np
from pubsub import pub
from pathlib import Path
from sortedcontainers import SortedDict
from scanbuddy.proc.fdata import ExtractFdata
from scanbuddy.proc.converter import Converter
from pydicom.tag import Tag
from scanbuddy.common.dicom import get_tag, get_path

logger = logging.getLogger(__name__)

class BoldProcessor:
    def __init__(self, config, debug_display=False):
        self.reset()
        self._config = config
        self._debug_display = self._config.find_one('$.app.debug_display', default=debug_display)
        # New: read SNR enable flag from config (default True)
        self._snr_enabled = self._config.find_one('$.app.snr.enabled', default=True)
        pub.subscribe(self.reset, 'reset')
        pub.subscribe(self.listener, 'bold-proc')

    def reset(self):
        self._instances = SortedDict()
        self._snr_instances = SortedDict()
        self.make_arrays_zero('reset')
        pub.sendMessage('plot_snr', snr_metric=str(0.0))
        logger.debug('received message to reset')

    def listener(self, ds, path, modality):
        logger.info('inside of the bold-proc topic')
        self._modality = modality
        key = int(ds.InstanceNumber)
        is_multi_echo, is_TE2 = self.check_echo(ds)
        if is_multi_echo is True and is_TE2 is False:
            os.remove(path)
            return
        if is_multi_echo:
            key = self.get_new_key(key)
            logger.info(f'new multi-echo key: {key}')
        self._instances[key] = {
            'path': path,
            'volreg': None,
            'nii_path': None
        }
        self._snr_instances[key] = {
            'path': path,
            'fdata_array': None,
            'mask_threshold': None,
            'mask': None
        }
        logger.debug('current state of instances')
        logger.debug(json.dumps(self._instances, default=list, indent=2))

        logger.info('instantiating and running converter')
        converter = Converter()
        converter.run(self._instances[key], modality, key)

        tasks = self.check_volreg(key)
        logger.debug('publishing message to volreg topic with the following tasks')
        logger.debug(json.dumps(tasks, indent=2))

        pub.sendMessage('volreg', tasks=tasks, modality=modality)
        logger.debug(f'publishing message to params topic')
        pub.sendMessage('params', ds=ds, modality=modality)

        logger.debug(f'after volreg')
        logger.debug(json.dumps(self._instances, indent=2))
        project = ds.get('StudyDescription', '[STUDY]')
        session = ds.get('PatientID', '[PATIENT]')
        scandesc = ds.get('SeriesDescription', '[SERIES]')
        scannum = ds.get('SeriesNumber', '[NUMBER]')
        subtitle_string = f'{project} • {session} • {scandesc} • {scannum}'
        if key < 6:
            logger.info(f'Scan info: Project: {project}, Session: {session}, Series: {scandesc}, Scan Number: {scannum}, Date & Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')        
        num_vols = get_tag(ds, (0x0020, 0x0105))
        if self._debug_display:
            pub.sendMessage('plot', instances=self._instances, subtitle_string=subtitle_string)
        elif num_vols and num_vols == key:
            pub.sendMessage('plot', instances=self._instances, subtitle_string=subtitle_string)

        fdata_tasks = self.check_fdata(key)

        fdata_extraction = ExtractFdata()
        nii_path = self._instances[key]['nii_path']
        fdata_extraction.do(nii_path, fdata_tasks)

        # Only publish SNR tasks if enabled
        if self._snr_enabled:
            snr_thread = threading.Thread(
                target=pub.sendMessage,
                args=('snr',),
                kwargs={'key': key, 'snr_instances': self._snr_instances, 'ds': ds}
            )
            snr_thread.start()
        else:
            logger.info('SNR disabled by configuration; skipping SNR task publish')

    def check_volreg(self, key):
        tasks = list()
        current = self._instances[key]

        i = self._instances.bisect_left(key)

        try:
            left_index = max(0, i - 1)
            left = self._instances.values()[left_index]
            logger.debug(f'to the left of {current["path"]} is {left["path"]}')
            tasks.append((current, left))
        except IndexError:
            pass

        try:
            right_index = i + 1
            right = self._instances.values()[right_index]
            logger.debug(f'to the right of {current["path"]} is {right["path"]}')
            tasks.append((right, current))
        except IndexError:
            pass

        return tasks

    def get_new_key(self, instance_number):
        return ((instance_number - 2) // 4) + 1

    def check_fdata(self, key):
        tasks = list()

        current_idx = self._snr_instances.bisect_left(key)

        try:
            value = self._snr_instances.values()[current_idx]
            tasks.append(value)
        except IndexError:
            pass

        return tasks

    def make_arrays_zero(self, moment='end'):
        if moment == 'end':
            logger.info('freeing up RAM from snr arrays')
        else:
            logger.debug('making sure snr arrays are deallocated')
        self._fdata_array = None
        self._slice_intensity_means = None


    def check_echo(self, ds):
        """
        Safely check for multi-echo / TE2 information.
        Returns (is_multi_echo, is_TE2).
        """
        seq_tag = Tag(0x5200, 0x9230)
        # Safely get the data element; ds.get returns None if missing
        elem = ds.get(seq_tag)
        if elem is None:
            return False, False

        # Extract sequence value if present
        seq = getattr(elem, "value", elem)
        if not seq:
            return False, False

        # Ensure we have at least one item
        try:
            first_item = seq[0]
        except Exception:
            return False, False

        # Look for Echo Time tag in items to detect multi-echo
        echo_time_tag = Tag(0x0018, 0x0081)  # (0018,0081) Echo Time
        is_multi_echo = len(seq) > 1
        is_TE2 = False
        if echo_time_tag in first_item:
            if is_multi_echo and echo_time_tag in seq[1]:
                is_TE2 = True
        else:
            # no echo time info — treat as single echo
            is_multi_echo = False
            is_TE2 = False

        return is_multi_echo, is_TE2

    def get_echo_time(self, ds):
        # Per-frame FG -> MR Timing and Related Parameters Sequence -> EchoTime
        return get_path(ds, (0x5200, 0x9230), 0, (0x0018, 0x9114), 0, (0x0018, 0x0081))
