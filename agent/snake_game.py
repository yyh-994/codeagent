#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贪吃蛇游戏 (美化版) - 使用 tkinter 实现弹窗运行
方向键控制，空格暂停，Q 退出
外观：渐变星空背景、圆角渐变色蛇身、发光食物、粒子特效、侧边计分面板
"""

import tkinter as tk
import math
import random

# ============ 游戏常量 ============
CELL_SIZE = 22        # 每个格子像素大小
GRID_WIDTH = 25       # 横向格子数（游戏区）
GRID_HEIGHT = 20      # 纵向格子数
PANEL_WIDTH = 180     # 侧边栏宽度

# 游戏区画布尺寸
AREA_WIDTH = CELL_SIZE * GRID_WIDTH
AREA_HEIGHT = CELL_SIZE * GRID_HEIGHT
WINDOW_WIDTH = AREA_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = AREA_HEIGHT

INIT_SPEED = 150      # 初始速度（毫秒）
MIN_SPEED = 60        # 最快速度

# ============ 配色主题 ============
BG_COLOR = "#0e1b2e"            # 深蓝背景
PANEL_BG = "#13263d"            # 侧边栏背景
GRID_LINE = "#1a3050"           # 网格线
PANEL_LINE = "#244265"          # 面板分隔线
TEXT_MAIN = "#e8f4ff"           # 主文字
TEXT_DIM = "#8fb3d9"            # 次要文字
ACCENT = "#4fd1ff"              # 强调色（青）
FOOD_COLOR = "#ff5d73"          # 食物红
FOOD_GLOW = "#ff8fa3"

# 蛇身渐变（尾部到头部）
SNAKE_TAIL = "#1fd1b3"
SNAKE_MID = "#36e0a3"
SNAKE_HEAD = "#8aff7a"
SNAKE_OUTLINE = "#0d3b33"


class SnakeGame:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("贪吃蛇")
        self.window.resizable(False, False)
        self.window.configure(bg=BG_COLOR)

        # 粒子特效（吃到食物时）
        self.particles = []

        # 游戏状态
        self.reset_game()

        # ---- 侧边栏（面板）----
        self.panel = tk.Frame(self.window, bg=PANEL_BG, width=PANEL_WIDTH)
        self.panel.pack(side="right", fill="both")
        self.panel.pack_propagate(False)

        # 标题
        tk.Label(
            self.panel, text="🐍 贪吃蛇", bg=PANEL_BG, fg=ACCENT,
            font=("Microsoft YaHei", 18, "bold"),
        ).pack(pady=(24, 6))

        tk.Label(
            self.panel, text="SNAKE", bg=PANEL_BG, fg=TEXT_DIM,
            font=("Arial", 10),
        ).pack(pady=(0, 18))

        self.divider(8)

        # 分数
        tk.Label(
            self.panel, text="当前得分", bg=PANEL_BG, fg=TEXT_DIM,
            font=("Microsoft YaHei", 10),
        ).pack(pady=(6, 0))
        self.score_value = tk.Label(
            self.panel, text="0", bg=PANEL_BG, fg=TEXT_MAIN,
            font=("Arial", 32, "bold"),
        )
        self.score_value.pack()

        self.divider(8)

        # 长度
        tk.Label(
            self.panel, text="蛇身长度", bg=PANEL_BG, fg=TEXT_DIM,
            font=("Microsoft YaHei", 10),
        ).pack(pady=(6, 0))
        self.length_value = tk.Label(
            self.panel, text="3", bg=PANEL_BG, fg=TEXT_MAIN,
            font=("Arial", 20, "bold"),
        )
        self.length_value.pack()

        self.divider(8)

        # 最高分
        self.high_score = 0
        tk.Label(
            self.panel, text="最高纪录", bg=PANEL_BG, fg=TEXT_DIM,
            font=("Microsoft YaHei", 10),
        ).pack(pady=(6, 0))
        self.high_score_value = tk.Label(
            self.panel, text="0", bg=PANEL_BG, fg=ACCENT,
            font=("Arial", 20, "bold"),
        )
        self.high_score_value.pack()

        # 操作提示（底部）
        self.help_frame = tk.Frame(self.panel, bg=PANEL_BG)
        self.help_frame.pack(side="bottom", pady=16)
        tips = [
            ("↑↓←→", "移动"),
            ("空格", "暂停"),
            ("Q", "退出"),
        ]
        for key, desc in tips:
            tk.Label(
                self.help_frame, text=f"{key}  {desc}", bg=PANEL_BG,
                fg=TEXT_DIM, font=("Microsoft YaHei", 10), justify="left",
            ).pack(anchor="w", pady=2)

        # ---- 游戏画布 ----
        self.canvas = tk.Canvas(
            self.window,
            width=AREA_WIDTH,
            height=AREA_HEIGHT,
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(side="left")

        # 密钥绑定
        self.window.bind("<Up>", lambda e: self.change_direction("up"))
        self.window.bind("<Down>", lambda e: self.change_direction("down"))
        self.window.bind("<Left>", lambda e: self.change_direction("left"))
        self.window.bind("<Right>", lambda e: self.change_direction("right"))
        self.window.bind("<space>", lambda e: self.toggle_pause())
        self.window.bind("q", lambda e: self.window.destroy())
        self.window.bind("Q", lambda e: self.window.destroy())
        window_center_x = self.window.winfo_screenwidth() // 2
        window_center_y = self.window.winfo_screenheight() // 2
        self.window.update_idletasks()
        self.window.geometry(
            f"+{window_center_x - WINDOW_WIDTH // 2}+{window_center_y - WINDOW_HEIGHT // 2}"
        )
        self.window.focus_set()

        # 生成星点背景
        self.stars = [
            (random.randint(0, AREA_WIDTH), random.randint(0, AREA_HEIGHT),
             random.randint(1, 2))
            for _ in range(60)
        ]

        # 开始游戏
        self.start_overlay = True
        self.window.after(INIT_SPEED, self.game_loop)
        self.window.mainloop()

    def divider(self, pad):
        """画一条分隔线"""
        frame = tk.Frame(self.panel, bg=PANEL_BG, height=pad)
        frame.pack(fill="both")
        tk.Frame(self.panel, bg=PANEL_LINE, height=1).pack(fill="x", padx=20)

    def reset_game(self):
        """初始化游戏状态"""
        cx = GRID_WIDTH // 2
        cy = GRID_HEIGHT // 2
        self.snake = [[cx - 2, cy], [cx - 1, cy], [cx, cy]]
        self.direction = "right"
        self.pending_direction = "right"
        self.food = None
        self.score = 0
        self.paused = False
        self.game_over = False
        self.speed = INIT_SPEED
        self.spawn_food()

    def spawn_food(self):
        """随机生成食物，避免与蛇重叠"""
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    def change_direction(self, new_dir):
        """记录待处理方向，阻止 180 度反向"""
        opposites = {
            "up": "down", "down": "up",
            "left": "right", "right": "left",
        }
        if new_dir != opposites[self.direction]:
            self.pending_direction = new_dir

    def toggle_pause(self):
        """暂停/继续"""
        if self.start_overlay:
            self.start_overlay = False
        elif not self.game_over:
            self.paused = not self.paused

    def game_loop(self):
        """主游戏循环"""
        if self.start_overlay:
            self.draw()
            self.window.after(120, self.game_loop)
            return

        if self.game_over:
            self.draw()
            self.update_particles()
            self.window.after(50, self.game_loop)
            return

        if not self.paused:
            self.step()

        self.draw()
        self.update_particles()
        self.window.after(self.speed, self.game_loop)

    def spawn_particles(self, cx, cy, color, n=18):
        """在 (cx,cy) 处生成爆炸粒子"""
        for _ in range(n):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4.5)
            self.particles.append({
                "x": cx, "y": cy,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": random.randint(20, 40),
                "color": color,
            })

    def update_particles(self):
        """更新并清理粒子"""
        alive = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vy"] += 0.08          # 轻微重力
            p["life"] -= 1
            if p["life"] > 0:
                alive.append(p)
        self.particles = alive

    def step(self):
        """推进一帧游戏逻辑"""
        self.direction = self.pending_direction
        head = self.snake[-1][:]
        dx, dy = 0, 0
        if self.direction == "up":
            dy = -1
        elif self.direction == "down":
            dy = 1
        elif self.direction == "left":
            dx = -1
        elif self.direction == "right":
            dx = 1

        new_head = [head[0] + dx, head[1] + dy]

        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH
                or new_head[1] < 0 or new_head[1] >= GRID_HEIGHT):
            self.game_over = True
            return

        if new_head in self.snake:
            self.game_over = True
            return

        self.snake.append(new_head)

        if new_head == self.food:
            self.score += 10
            if self.score > self.high_score:
                self.high_score = self.score
            self.speed = max(MIN_SPEED, self.speed - 4)
            # 食物爆炸粒子
            fx = new_head[0] * CELL_SIZE + CELL_SIZE / 2
            fy = new_head[1] * CELL_SIZE + CELL_SIZE / 2
            self.spawn_particles(fx, fy, FOOD_COLOR)
            self.spawn_particles(fx, fy, "#ffe08a", 10)
            self.spawn_food()
        else:
            self.snake.pop(0)

    def draw(self):
        """绘制画面"""
        self.canvas.delete("all")

        # ---- 星空渐变背景 ----
        for i in range(AREA_HEIGHT):
            ratio = i / AREA_HEIGHT
            r = int(0x0e + ratio * 12)
            g = int(0x1b + ratio * 18)
            b = int(0x2e + ratio * 28)
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, AREA_WIDTH, i, fill=color)

        # 星星
        for sx, sy, size in self.stars:
            self.canvas.create_oval(
                sx, sy, sx + size, sy + size, fill="#cfe8ff", outline=""
            )

        # ---- 网格线 ----
        for x in range(0, AREA_WIDTH, CELL_SIZE):
            self.canvas.create_line(x, 0, x, AREA_HEIGHT, fill=GRID_LINE)
        for y in range(0, AREA_HEIGHT, CELL_SIZE):
            self.canvas.create_line(0, y, AREA_WIDTH, y, fill=GRID_LINE)

        # ---- 蛇身（圆角，渐变色）----
        n = len(self.snake)
        for i, (sx, sy) in enumerate(self.snake):
            x1 = sx * CELL_SIZE + 2
            y1 = sy * CELL_SIZE + 2
            x2 = sx * CELL_SIZE + CELL_SIZE - 2
            y2 = sy * CELL_SIZE + CELL_SIZE - 2

            # 颜色插值：尾部 -> 头部
            t = i / max(1, n - 1)
            color = self._lerp_color(SNAKE_TAIL, SNAKE_MID, t)
            if i == n - 1:
                color = SNAKE_HEAD

            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=color, outline=SNAKE_OUTLINE,
                width=1,
            )

        # ---- 食物（带发光效果）----
        if self.food:
            fx, fy = self.food
            cx = fx * CELL_SIZE + CELL_SIZE / 2
            cy = fy * CELL_SIZE + CELL_SIZE / 2
            # 外圈光晕
            self.canvas.create_oval(
                cx - 9, cy - 9, cx + 9, cy + 9, fill=FOOD_GLOW, outline="", stipple="gray50"
            )
            self.canvas.create_oval(
                cx - 7, cy - 7, cx + 7, cy + 7, fill=FOOD_COLOR, outline="#ffe08a", width=1
            )
            # 高光
            self.canvas.create_oval(
                cx - 3, cy - 3.5, cx + 1, cy + 0.5, fill="white", outline=""
            )

        # ---- 粒子特效 ----
        for p in self.particles:
            alpha = p["life"] / 40
            size = int(2 + alpha * 2)
            self.canvas.create_oval(
                p["x"], p["y"], p["x"] + size, p["y"] + size,
                fill=p["color"], outline="", stipple="gray50",
            )

        # ---- 侧边栏更新 ----
        self.score_value.config(text=str(self.score))
        self.length_value.config(text=str(len(self.snake)))
        self.high_score_value.config(text=str(self.high_score))

        # ---- 覆盖层文字 ----
        if self.start_overlay:
            self._center_text("🐍 贪吃蛇", 40, ACCENT, y_ratio=0.42)
            self._center_text("方向键移动 · 空格开始", 16, TEXT_DIM, y_ratio=0.5)
            self._center_text("按 空格 开始游戏", 14, TEXT_MAIN, y_ratio=0.58)
        elif self.game_over:
            self._center_text("游戏结束", 44, FOOD_COLOR, y_ratio=0.4)
            self._center_text(f"得分: {self.score}", 22, TEXT_MAIN, y_ratio=0.5)
            self._center_text("按 空格 重新开始", 14, TEXT_DIM, y_ratio=0.6)

    def _center_text(self, text, size, color, y_ratio, y_offset=0):
        """居中绘制文字（带黑底遮罩增加可读性）"""
        x = AREA_WIDTH // 2
        y = AREA_HEIGHT * y_ratio + y_offset
        self.canvas.create_text(
            x, y, text=text, fill=color,
            font=("Microsoft YaHei", size, "bold"),
        )

    def _lerp_color(self, c1, c2, t):
        """在两个 hex 颜色之间线性插值"""
        def hx(h):
            return int(h, 16)
        r1, g1, b1 = hx(c1[1:3]), hx(c1[3:5]), hx(c1[5:7])
        r2, g2, b2 = hx(c2[1:3]), hx(c2[3:5]), hx(c2[5:7])
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    SnakeGame()
