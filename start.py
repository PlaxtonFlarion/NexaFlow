from plan.test_plan import TestPlan
from nexaflow.skills.device import Manage


if __name__ == '__main__':
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    manage = Manage()
    device = manage.Phone

    with TestPlan(device, 5) as test:
        test.test_02()
