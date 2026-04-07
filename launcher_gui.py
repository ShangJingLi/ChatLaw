import os
import sys
import re
import subprocess

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QMessageBox
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_PATH = os.path.join(PROJECT_ROOT, "launcher.py")


def is_valid_ip(ip: str) -> bool:
    pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if not re.match(pattern, ip):
        return False
    parts = ip.split(".")
    return all(0 <= int(p) <= 255 for p in parts)


def start_client(server_ip: str):
    if not os.path.exists(LAUNCHER_PATH):
        raise FileNotFoundError(f"未找到 launcher.py: {LAUNCHER_PATH}")

    # 给子进程补上项目根目录
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + old_pythonpath
    else:
        env["PYTHONPATH"] = PROJECT_ROOT

    # 调试阶段：用 cmd /k，出错后终端不会自动关闭
    cmd = [
        "cmd", "/k",
        sys.executable,
        LAUNCHER_PATH,
        "client",
        "pt",
        "--SERVER_IP",
        server_ip
    ]

    print(f"[LauncherGUI] Execute: {cmd}", flush=True)
    print(f"[LauncherGUI] PROJECT_ROOT: {PROJECT_ROOT}", flush=True)
    print(f"[LauncherGUI] PYTHONPATH: {env['PYTHONPATH']}", flush=True)

    subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        env=env
    )


class Launcher(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ChatLaw 启动器")
        self.setFixedSize(420, 180)

        layout = QVBoxLayout()

        self.label = QLabel("请输入服务器 IP")
        layout.addWidget(self.label)

        self.input = QLineEdit()
        self.input.setPlaceholderText("例如：47.94.149.246")
        layout.addWidget(self.input)

        self.button = QPushButton("启动 ChatLaw")
        self.button.clicked.connect(self.on_confirm)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def on_confirm(self):
        server_ip = self.input.text().strip()

        if not server_ip:
            QMessageBox.warning(self, "提示", "请输入服务器 IP")
            return

        if not is_valid_ip(server_ip):
            QMessageBox.warning(self, "提示", "IP 地址格式不正确")
            return

        try:
            self.button.setEnabled(False)
            self.button.setText("正在启动...")
            QApplication.processEvents()

            start_client(server_ip)
            self.close()

        except Exception as e:
            self.button.setEnabled(True)
            self.button.setText("启动 ChatLaw")
            QMessageBox.critical(self, "启动失败", str(e))


def main():
    # 补回项目根目录，防止当前进程自己导入时也出问题
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)

    app = QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()