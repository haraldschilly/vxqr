#  VXQR1
#
#  Copyright 2015  Harald Schilly <harald.schilly@univie.ac.at>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import logging
import resource


def clock_user_sys():
    u, s = resource.getrusage(resource.RUSAGE_SELF)[:2]
    return u + s


class ColoredFormatter(logging.Formatter):

    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[0;%dm"
    COLOR_SEQ_BOLD = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        'DEBUG': BLUE,
        'INFO': WHITE,
        'WARNING': YELLOW,
        'CRITICAL': MAGENTA,
        'ERROR': RED
    }

    def __init__(self):
        msg = '%(runtime).2f $BOLD%(name)-2s %(where)-12s$RESET %(levelname)-5s %(message)s'
        msg = msg\
            .replace("$RESET", ColoredFormatter.RESET_SEQ)\
            .replace("$BOLD", ColoredFormatter.BOLD_SEQ)
        logging.Formatter.__init__(self, fmt=msg)

    @staticmethod
    def colorize(string, color, bold=False):
        cs = ColoredFormatter.COLOR_SEQ_BOLD if bold else ColoredFormatter.COLOR_SEQ
        string = '%s%s%s' % (
            cs % (30 + color), string, ColoredFormatter.RESET_SEQ)
        string = "%-20s" % string
        return string

    def format(self, record):
        levelname = record.levelname
        if levelname in ColoredFormatter.COLORS:
            col = ColoredFormatter.COLORS[levelname]
            record.name = self.colorize(record.name, col, True)
            record.lineno = self.colorize(record.lineno, col, True)
            record.levelname = self.colorize(levelname, col, True)
            record.msg = self.colorize(record.msg, col)
        return logging.Formatter.format(self, record)


class VXQR1LoggingContext(logging.Filter):

    def __init__(self):
        logging.Filter.__init__(self)
        self._start = clock_user_sys()

    def filter(self, record):
        record.runtime = clock_user_sys() - self._start
        record.where = "%s:%s" % (record.filename[:-3], record.lineno)
        return True

_vxlogger = {}

# NOTE this has to return the logger object and do not wrap it into a class.
# Otherwise the context doesn't know the file and line number!


def create_logger(name, level=logging.INFO):
    """
    Creates logger with ``name`` and given ``level`` logging level.
    """
    global _vxlogger
    if name in _vxlogger:
        logger = _vxlogger[name]
    else:
        logger = logging.getLogger(name)
        logger.addFilter(VXQR1LoggingContext())
        log_stream_handler = logging.StreamHandler()
        log_stream_handler.setLevel(logging.DEBUG)
        log_formatter = ColoredFormatter()
        log_stream_handler.setFormatter(log_formatter)
        logger.addHandler(log_stream_handler)
        _vxlogger[name] = logger
    logger.setLevel(level)
    return logger


class VXQR1Exception(Exception):

    def __init__(self, msg):
        super(VXQR1Exception, self).__init__(msg)
