# setup.py
import setuptools
from pathlib import Path

# README.md 를 long_description 으로 쓰려면
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="peetsModules",                             # PyPI에 올릴 때 사용할 패키지 이름
    version="0.0.1",                                  # 배포 버전
    author="Glaysia",                                 # 작성자 이름
    author_email="willbecat27@gmail.com",            # 작성자 이메일
    description="전력전자 자동화에 필요한 유틸리티 모듈 모음",  # 한줄 설명
    long_description=long_description,                # README.md 전체를 설명으로 사용
    long_description_content_type="text/markdown",    # README.md 포맷
    url="https://github.com/Glaysia/peetsModules",    # 프로젝트 URL
    packages=setuptools.find_packages("peetsModules"),              # 모든 서브패키지를 자동으로 포함
    package_dir={"":"peetsModules"},
    python_requires=">=3.10",                          # 지원하는 Python 버전
    install_requires=[                                # pip install 시 자동으로 설치할 의존 패키지
        # 예시:
        # "numpy>=1.20",
        # "pandas>=1.2",
    ],
    classifiers=[                                     # PyPI 메타데이터
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: Microsoft :: Windows :: Windows 11"
    ],
    include_package_data=True,                        # MANIFEST.in 에 추가한 리소스도 포함
)
