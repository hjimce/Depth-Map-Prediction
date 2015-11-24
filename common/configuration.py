#coding=utf-8
'''
Copyright (C) 2014 New York University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import importlib

from ConfigParser import SafeConfigParser, NoOptionError, NoSectionError

def read_config(fn):
    conf = _ConfigParser()
    conf.read(fn)
    conf.set_eval_environ(section='config')
    return conf

_ERROR = object()

class _ConfigParser(SafeConfigParser):
    def __init__(self):
        SafeConfigParser.__init__(self)
        self.eval_globals = None
        self.eval_locals = None

    def get_section(self, section):
        return _ConfigSection(self, section)

    def set_eval_environ(self, section=None, globals=None, locals=None):
        self.eval_globals = globals or {}
        self.eval_locals = locals
        self.eval_globals.update(self._read_eval_env(section))

    def _read_eval_env(self, section):
        if not section or not self.has_section(section):
            return {}
        mods = self.get(section, 'imports', '')
        eval_env = {}
        for modstr in mods.split(','):
            if ' as ' in modstr:
                (mod, name) = modstr.split(' as ')
            else:
                mod = name = modstr
            eval_env[name.strip()] = importlib.import_module(mod.strip())
        return eval_env

    def get_eval_environ(self, globals, locals):
        if globals is None:
            globals = self.eval_globals
        if locals is None:
            locals = self.eval_locals
        return (globals, locals)

    def geteval(self, section, option,
                default=_ERROR, globals=None, locals=None):
        (globals, locals) = self.get_eval_environ(globals, locals)
        if isinstance(section, (tuple, list)):
            for sec in section:
                try:
                    return self.geteval(sec, option, _ERROR, globals, locals)
                except (NoOptionError, NoSectionError), ex:
                    pass
            if default is not _ERROR:
                return default
            raise ex
        try:
            return eval(self.get(section, option), globals, locals)
        except NoOptionError:
            if default is not _ERROR:
                return default
            raise

    def __get(self, section, option, default, getf):
        if isinstance(section, (tuple, list)):
            for sec in section:
                try:
                    return self.__get(sec, option, _ERROR, getf)
                except (NoOptionError, NoSectionError), ex:
                    pass
            if default is not _ERROR:
                return default
            raise ex
        try:
            return getf(self, section, option)
        except NoOptionError:
            if default is not _ERROR:
                return default
            raise

    def get(self, section, option, default=_ERROR):
        return self.__get(section, option, default, SafeConfigParser.get)

    def getint(self, section, option, default=_ERROR):
        return self.__get(section, option, default, SafeConfigParser.getint)

    def getfloat(self, section, option, default=_ERROR):
        return self.__get(section, option, default, SafeConfigParser.getfloat)

    def getboolean(self, section, option, default=_ERROR):
        return self.__get(section, option, default, SafeConfigParser.getboolean)

class _ConfigSection(object):
    def __init__(self, conf, section):
        self.conf = conf
        self.parent = conf
        self.section = section
        self.eval_globals = None
        self.eval_locals = None

    def set_eval_environ(self, section=None, globals=None, locals=None):
        self.eval_globals = globals or {}
        self.eval_locals = locals
        self.eval_globals.update(self.conf._read_eval_env(section))

    def get_eval_environ(self, globals, locals):
        if globals is None:
            globals = self.eval_globals
            if globals is None:
                globals = self.conf.eval_globals
        if locals is None:
            locals = self.eval_locals
            if locals is None:
                locals = self.conf.eval_locals
        return (globals, locals)

    def geteval(self, option, default=_ERROR, globals=None, locals=None):
        (globals, locals) = self.get_eval_environ(globals, locals)
        try:
            return eval(self.get(option), globals, locals)
        except NoOptionError:
            if default is not _ERROR:
                return default
            raise

    def __getattr__(self, option):
        val = self.conf.get(self.section, option)

    def has_option(self, *args):
        return self.conf.has_option(self.section, *args)

    def get(self, option, default=_ERROR):
        return self.conf.get(self.section, option, default)

    def getint(self, option, default=_ERROR):
        return self.conf.getint(self.section, option, default)

    def getfloat(self, option, default=_ERROR):
        return self.conf.getfloat(self.section, option, default)

    def getboolean(self, option, default=_ERROR):
        return self.conf.getboolean(self.section, option, default)

    def items(self, *args):
        return self.conf.items(self.section, *args)

    def set(self, *args):
        return self.conf.set(self.section, *args)

    def remove_option(self, *args):
        return self.conf.remove_option(self.section, *args)
