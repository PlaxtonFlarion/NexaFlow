import time
from rich.console import Console
from rich.progress import Progress

title = "讲个笑话"
a = [1, 2, 3, 4, 5]

with Progress() as progress:
    task_sheet = progress.add_task(f"[bold #FFFFD7]{title}", total=len(a))
    while not progress.finished:
        progress.update(task_sheet, advance=1)
        time.sleep(5)




