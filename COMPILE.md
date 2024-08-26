# Framix(画帧秀) 编译 / Compile

![LOGO](resources/images/illustration/Compile.png)

---

## 前提条件

在开始之前，请确保已完成以下操作：

- 安装 [Python](https://www.python.org/downloads/) 3.11 或更高版本
- 安装 [Nuitka](https://nuitka.net/)
  - 导航到您的 Python 脚本所在的目录。
    ```
    pip install nuitka
    ```
- 确保在项目根目录下有一个 `requirements.txt` 文件，其中列出了所有的依赖包
> **NexaFlow**
>> **requirements.txt**
- 确保您的 Python 环境中安装了所有依赖包
  - 导航到您的 Python 脚本所在的目录。
    ```
    pip install -r requirements.txt
    ```
- 在 Python 脚本所在的目录新建 applications 文件夹
> **NexaFlow**
>> **applications**

---

## Windows
### 准备工作
- 打开命令提示符（Command Prompt）或 PowerShell。 
- 导航到您的 Python 脚本所在的目录。

### 运行 **Nuitka** 命令
```
python -m nuitka --standalone --windows-icon-from-ico=resources/icons/framix_icn_2.ico --nofollow-import-to=tensorflow,uiautomator2 --include-module=pdb,deprecation,xmltodict --include-package=ml_dtypes,distutils,site,google,absl,wrapt,gast,astunparse,termcolor,opt_einsum,flatbuffers,h5py,adbutils,apkutils2,cigam,pygments --show-progress --show-memory --output-dir=applications frameflow/framix.py
```

### 目录结构
- **applications**
  - **framix.dist**
    - **schematic**
      - **resources**
      - **supports**
      - **templates**
    - **...**
  - **Specially**

---

## MacOS
### 准备工作
- 打开终端（Terminal）。 
- 导航到您的 Python 脚本所在的目录。

### 运行 **Nuitka** 命令
```
python -m nuitka --standalone --macos-create-app-bundle --macos-app-icon=resources/icons/framix_icn_2.png --nofollow-import-to=tensorflow,uiautomator2 --include-module=pdb,deprecation,xmltodict --include-package=ml_dtypes,distutils,site,google,absl,wrapt,gast,astunparse,termcolor,opt_einsum,flatbuffers,h5py,adbutils,apkutils2,cigam,pygments --show-progress --show-memory --output-dir=applications frameflow/framix.py
```

### 目录结构
- **applications**
  - **framix.app**
    - **Contents**
      - **_CodeSignature**
      - **MacOS**
        - **schematic**
          - **resources**
          - **supports**
          - **templates**
        - **...**
      - **Resources**
      - **Specially**
      - **Info.plist**

---