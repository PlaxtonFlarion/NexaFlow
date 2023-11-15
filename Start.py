import asyncio
from plan.test_plan import TestPlan
from nexaflow.skills.device import Manage


async def main():
    # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
    manage = Manage()
    device = manage.Phone

    async with TestPlan(device, 5) as test:
        await test.test_02()


if __name__ == '__main__':
    asyncio.run(main())
