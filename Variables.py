import re
from typing import Union, Dict
from dataclasses import dataclass
from pathlib import Path
from os.path import join


@dataclass
class Variable(dict):
    name: str
    vtype: str
    io: str
    value: Union[str, float, bool, None]

class Variables(dict):
    _instance =None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance:dict = super().__new__(cls)
            dict.__init__(cls._instance)

            from GenericPythonStructs import PHXScriptWrapperObject
            cls.WRAPPERPATH = kwargs["initializer"]                 # "전처리부.scriptWrapper"
            cls.WRAPPEROBJ = PHXScriptWrapperObject(cls.WRAPPERPATH)
            # cls.PATH = cls.WRAPPEROBJ.getRunDirectory()             # C:\Users\harry\AppData\Roaming\Phoenix Integration\MCRE\peetsMBSE2\scripts
            # cls.WRAPPERPATH = join(cls.PATH, cls.WRAPPERPATH)       # C:\Users\harry\AppData\Roaming\Phoenix Integration\MCRE\peetsMBSE2\scripts\전처리부.scriptWrapper


            # cls._instance.update(
            #     cls.load_variables(cls.wrapper_full)  # 딱 한 번만 dict 생성
            # )

        # cls.update_cls()
        
        return cls._instance
    
    def update_cls(self):
        self.variables = {
            name: Variable(
                name=name,
                vtype=var.vtype,
                io=var.io,
                value=self.get(name)
            )
            for name, var in self.variables.items()
        }

    

    @classmethod
    def load_variables(
        cls,
        filepath: Union[str, Path]
    ) -> Dict[str, Variable]:
        """
        스크립트 파일에서 변수 선언문을 읽어
        변수명(name)을 키로, Variable 객체를 값으로 하는 dict를 반환.

        variable: <name> <type> <input|output> … default=<value> …
        """
        var_pattern = re.compile(
            r"""^\s*variable:\s*
            (?P<name>\w+)\s+
            (?P<type>\w+)\s+
            (?P<io>input|output)
            """,
            re.IGNORECASE | re.VERBOSE,
        )
        default_pattern = re.compile(r"default=([^\s]+)")

        variables: Dict[str, Variable] = {}

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.lstrip().startswith(("script:", "def run")):
                    break

                m = var_pattern.match(line)
                if not m:
                    continue

                name  = m.group("name")
                vtype = m.group("type")
                io    = m.group("io").lower()

                # default= 값 파싱 (숫자면 float, 아니면 string)
                default_m = default_pattern.search(line)
                if default_m:
                    raw = default_m.group(1).strip('"')
                    try:
                        value: Union[float, str] = float(raw)
                    except ValueError:
                        value = raw
                else:
                    value = None

                variables[name] = Variable(
                    name=name,
                    vtype=vtype,
                    io=io,
                    value=value,
                )

        return variables
