from plan.test_plan import TestPlan
from nexaflow.constants import Constants
from nexaflow.skills.device import Manage


if __name__ == '__main__':
    # pip freeze > requirements.txt
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tensorflow==2.14.0
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade tensorflow
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    Constants.initial_logger()
    manage = Manage()
    device = manage.Phone

    with TestPlan(device, 5) as test:
        test.test_02()
