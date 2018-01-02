import os
import time
from datetime import datetime
from getpass import getuser
from socket import gethostname
from threading import Thread

import requests
from six.moves.queue import Empty, Queue

from torchpack.runner.hooks import Hook


class PaviLoggerHook(Hook):

    def __init__(self,
                 interval,
                 url,
                 username=None,
                 password=None,
                 instance_id=None,
                 reset_meter=True,
                 ignore_last=True):
        self.interval = interval
        self.url = url
        self.username = self._get_env_var(username, 'PAVI_USERNAME')
        self.password = self._get_env_var(password, 'PAVI_PASSWORD')
        self.instance_id = instance_id
        self.reset_meter = reset_meter
        self.ignore_last = ignore_last
        self.log_queue = None

    def _get_env_var(self, var, env_var):
        if var is not None:
            return str(var)

        var = os.getenv(env_var)
        if not var:
            raise ValueError(
                '"{}" is neither specified nor defined as env variables'.
                format(env_var))
        return var

    def connect(self, model_name, work_dir=None, info=dict(), timeout=5):
        print('connecting pavi service {}...'.format(self.url))
        post_data = dict(
            time=str(datetime.now()),
            username=self.username,
            password=self.password,
            instance_id=self.instance_id,
            model=model_name,
            work_dir=os.path.abspath(work_dir) if work_dir else '',
            session_file=info.get('session_file', ''),
            session_text=info.get('session_text', ''),
            model_text=info.get('model_text', ''),
            device='{}@{}'.format(getuser(), gethostname()))
        try:
            response = requests.post(self.url, json=post_data, timeout=timeout)
        except Exception as ex:
            print('fail to connect to pavi service: {}'.format(ex))
        else:
            if response.status_code == 200:
                self.instance_id = response.text
                print('pavi service connected, instance_id: {}'.format(
                    self.instance_id))
                self.log_queue = Queue()
                self.log_thread = Thread(target=self.post_log)
                self.log_thread.daemon = True
                self.log_thread.start()
                return True
            else:
                print('fail to connect to pavi service, status code: '
                      '{}, err message: {}'.format(response.status_code,
                                                   response.reason))
        return False

    def post_log(self, max_retry=3, queue_timeout=1, req_timeout=3):
        while True:
            try:
                log = self.log_queue.get(timeout=queue_timeout)
            except Empty:
                time.sleep(1)
            except Exception as ex:
                print('fail to get logs from queue: {}'.format(ex))
            else:
                retry = 0
                while retry < max_retry:
                    try:
                        response = requests.post(
                            self.url, json=log, timeout=req_timeout)
                    except Exception as ex:
                        retry += 1
                        print('error when posting logs to pavi: {}'.format(ex))
                    else:
                        status_code = response.status_code
                        if status_code == 200:
                            break
                        else:
                            print('unexpected status code: %d, err msg: %s',
                                  status_code, response.reason)
                            retry += 1
                if retry == max_retry:
                    print('fail to send logs of iteration %d', log['iter_num'])

    def _log(self, runner):
        if self.log_queue is not None:
            log_outs = {
                var: runner.meter.avg[var]
                for var in runner.outputs['log_vars']
            }
            logs = {
                'time': str(datetime.now()),
                'instance_id': self.instance_id,
                'flow_id': runner.mode,
                'iter_num': runner.num_iters,
                'outputs': log_outs,
                'msg': ''
            }
            self.log_queue.put(logs)
            if self.reset_meter:
                runner.meter.reset()

    def after_train_iter(self, runner):
        if not self.every_n_inner_iters(runner, self.interval):
            if not self.end_of_epoch(runner):
                return
            elif self.ignore_last:
                return
        self._log(runner)

    def after_val_epoch(self, runner):
        self._log(runner)
