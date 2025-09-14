import tkinter as tk
from tkinter import ttk, messagebox
import threading
import subprocess
import sys
import os
import csv
from typing import List, Dict, Any

# Project-relative imports
try:
    from evaluation_metrics import evaluate_system_performance
except Exception:
    evaluate_system_performance = None  # Will fallback to subprocess

# ---------- Presenter Config (edit before demo) ----------
TEAM_NAME = "Команда: Decentarton"
TEAM_MEMBERS = [
    "Рустам — Team Lead / Data",
    "Аналитик — Product Logic",
    "Разработчик — Backend/NLG"
]
SOLUTION_SLOGAN = "Персональные пуши на основе реального поведения клиента"

# Paths (assuming we run from src/)
OUTPUT_CSV = os.path.join("..", "data", "output.csv")
CLIENTS_CSV = os.path.join("..", "data", "clients.csv")


class DemoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Demo: Персонализация банковских предложений")
        self.geometry("1080x720")
        self.minsize(960, 640)
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        # Top controls
        controls = ttk.Frame(container)
        controls.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        run_btn = ttk.Button(controls, text="1) Запустить пайплайн", command=self._run_pipeline_async)
        eval_btn = ttk.Button(controls, text="2) Оценка качества", command=self._run_evaluation_async)
        examples_btn = ttk.Button(controls, text="Показать примеры", command=self._show_examples)
        open_csv_btn = ttk.Button(controls, text="Открыть output.csv", command=self._open_output_csv)
        exit_btn = ttk.Button(controls, text="Выход", command=self.destroy)

        run_btn.pack(side=tk.LEFT, padx=(0, 6))
        eval_btn.pack(side=tk.LEFT, padx=6)
        examples_btn.pack(side=tk.LEFT, padx=6)
        open_csv_btn.pack(side=tk.LEFT, padx=6)
        exit_btn.pack(side=tk.RIGHT)

        # Main content with Notebook
        self.nb = ttk.Notebook(container)
        self.nb.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        self.slides: Dict[str, tk.Widget] = {}
        self._add_slide("1. Титульный", self._slide_title())
        self._add_slide("2. Проблема и цель", self._slide_problem_goal())
        self._add_slide("3. Данные", self._slide_data())
        self._add_slide("4. Подход", self._slide_approach())
        self._add_slide("5. Генерация пушей", self._slide_nlg())
        self._add_slide("6. Примеры", self._slide_examples())
        self._add_slide("7. Метрики", self._slide_metrics())
        self._add_slide("8–9. Итоги", self._slide_conclusion())

        # Log pane
        log_frame = ttk.LabelFrame(container, text="Логи / Ход демонстрации")
        log_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(log_frame, height=8, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=yscroll.set)

    def _add_slide(self, title: str, frame: tk.Widget) -> None:
        self.nb.add(frame, text=title)
        self.slides[title] = frame

    # ---------- Slides ----------
    def _slide_title(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        ttk.Label(f, text=TEAM_NAME, font=("Segoe UI", 20, "bold")).pack(anchor="w", padx=16, pady=(16, 4))
        ttk.Label(f, text="Участники:", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=16)
        for m in TEAM_MEMBERS:
            ttk.Label(f, text=f"• {m}", font=("Segoe UI", 11)).pack(anchor="w", padx=24)
        ttk.Label(f, text=SOLUTION_SLOGAN, font=("Segoe UI", 12), foreground="#1a73e8").pack(anchor="w", padx=16, pady=(12, 16))
        return f

    def _slide_problem_goal(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        self._bullets(f, "Почему одинаковые пуши не работают", [
            "Усталость от нерелевантных сообщений",
            "Низкая конверсия, упущенная выгода (кешбэк/проценты)"
        ], top=16)
        self._bullets(f, "Цель", [
            "Персонализировать уведомления по поведению за 3 месяца",
            "Рассчитать ожидаемую выгоду по каждому продукту",
            "Выбрать лучший продукт, сгенерировать понятный пуш"
        ], top=12)
        return f

    def _slide_data(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        self._bullets(f, "Источники", [
            "Профиль клиента: код, имя, статус, город, баланс",
            "Транзакции 3 месяца: дата, категория, сумма, валюта",
            "Переводы 3 месяца: тип, направление, сумма, валюта"
        ], top=16)
        self._bullets(f, "Объём", [
            "60 клиентов × 3 месяца"
        ], top=12)
        self._bullets(f, "Предобработка", [
            "Очистка, агрегация трат/переводов",
            "RFM-D признаки, кластеризация, propensity"
        ], top=12)
        return f

    def _slide_approach(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        self._bullets(f, "Сигналы выгоды (примеры)", [
            "Карта для путешествий: 4% на путешествия и такси",
            "Премиальная: 2/3/4% + лимит кешбэка 100k, 4% на рестораны/ювелирку",
            "Кредитная: 10% в топ‑3 категориях + 10% онлайн (игры/доставка/кино)",
            "Депозиты: 16.5 / 15.5 / 14.5% годовых"
        ], top=16)
        self._bullets(f, "Выбор продукта", [
            "Рассчёт выгод по всем продуктам",
            "Взвешенный скоринг (accuracy‑фокус) + мягкая диверсификация",
            "Правила верхнего уровня для редких/премиум кейсов"
        ], top=12)
        return f

    def _slide_nlg(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        self._bullets(f, "Структура пуша", [
            "Контекст клиента → конкретная польза → CTA",
            "Длина 180–220 символов, дружелюбный TOV",
            "Шаблоны Jinja + автоматическая подстановка чисел"
        ], top=16)
        return f

    def _slide_examples(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        ttk.Label(f, text="Примеры (client → product → push)", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=16, pady=(16, 8))
        self.examples_container = ttk.Frame(f)
        self.examples_container.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._render_examples_cards(self.examples_container, read_examples_from_csv(OUTPUT_CSV, limit=3))
        return f

    def _slide_metrics(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        ttk.Label(f, text="Метрики качества (из evaluation)", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=16, pady=(16, 8))
        self.metrics_text = tk.Text(f, height=14, wrap="word")
        self.metrics_text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.metrics_text.insert("end", "Нажмите ‘Оценка качества’ для актуального отчёта…")
        self.metrics_text.configure(state="disabled")
        return f

    def _slide_conclusion(self) -> ttk.Frame:
        f = ttk.Frame(self.nb)
        self._bullets(f, "Итоговое решение", [
            "CSV: client_code, product, push_notification",
            "Сильные стороны: точность, читабельный TOV, автоматизация",
            "Масштабирование: A/B‑тесты, сегментация, расширение продуктовой линейки"
        ], top=16)
        self._bullets(f, "Вывод", [
            "Персонализация поднимает интерес клиента и выгоду банка"
        ], top=12)
        return f

    def _bullets(self, parent: tk.Widget, title: str, items: List[str], top: int = 8) -> None:
        ttk.Label(parent, text=title, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=16, pady=(top, 6))
        for it in items:
            ttk.Label(parent, text=f"• {it}", font=("Segoe UI", 11)).pack(anchor="w", padx=24)

    # ---------- Actions ----------
    def _log(self, msg: str) -> None:
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def _run_pipeline_async(self) -> None:
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self) -> None:
        self._log("[RUN] Запуск main.py…")
        try:
            proc = subprocess.run([sys.executable, "main.py"], cwd=os.path.dirname(__file__), capture_output=True, text=True)
            self._log(proc.stdout.strip() or "(stdout пуст)")
            if proc.stderr:
                self._log("[stderr] " + proc.stderr.strip())
            if proc.returncode == 0 and os.path.exists(OUTPUT_CSV):
                self._log("[OK] Пайплайн завершён. Обновлён output.csv")
                # Refresh examples
                self._render_examples_cards(self.examples_container, read_examples_from_csv(OUTPUT_CSV, limit=3))
            else:
                self._log("[ERR] Пайплайн завершился с ошибкой или нет output.csv")
                messagebox.showerror("Ошибка", "Пайплайн завершился с ошибкой или отсутствует output.csv")
        except Exception as e:
            self._log(f"[EXC] {e}")
            messagebox.showerror("Исключение", str(e))

    def _run_evaluation_async(self) -> None:
        threading.Thread(target=self._run_evaluation, daemon=True).start()

    def _run_evaluation(self) -> None:
        self._log("[EVAL] Оценка качества…")
        try:
            metrics_text = None
            if evaluate_system_performance is not None:
                metrics = evaluate_system_performance(OUTPUT_CSV, CLIENTS_CSV)
                metrics_text = self._format_metrics(metrics)
            else:
                proc = subprocess.run([sys.executable, "evaluation_metrics.py", "--report"], cwd=os.path.dirname(__file__), capture_output=True, text=True)
                self._log(proc.stdout.strip() or "(stdout пуст)")
                if proc.stderr:
                    self._log("[stderr] " + proc.stderr.strip())
                metrics_text = "Отчёт сгенерирован (см. data/output_evaluation_report.md)."

            if metrics_text:
                self.metrics_text.configure(state="normal")
                self.metrics_text.delete("1.0", "end")
                self.metrics_text.insert("end", metrics_text)
                self.metrics_text.configure(state="disabled")
                self._log("[OK] Оценка завершена.")
        except Exception as e:
            self._log(f"[EXC] {e}")
            messagebox.showerror("Исключение", str(e))

    def _format_metrics(self, m: Dict[str, Any]) -> str:
        if not m:
            return "Нет метрик."
        lines = []
        lines.append("Основные метрики (по ТЗ):")
        lines.append(f"- Точность продукта: {m.get('product_accuracy', 0):.1f}/20")
        lines.append(f"- Качество пуша: {m.get('push_quality', 0):.1f}/20")
        lines.append(f"- Общий балл: {m.get('total_score', 0):.1f}/40 ({m.get('percentage', 0):.1f}%)")
        lines.append(f"- Клиентов оценено: {m.get('clients_evaluated', 0)}")
        return "\n".join(lines)

    def _open_output_csv(self) -> None:
        try:
            if os.name == 'nt':
                os.startfile(os.path.abspath(OUTPUT_CSV))  # type: ignore[attr-defined]
            elif sys.platform == 'darwin':
                subprocess.call(["open", os.path.abspath(OUTPUT_CSV)])
            else:
                subprocess.call(["xdg-open", os.path.abspath(OUTPUT_CSV)])
        except Exception as e:
            self._log(f"[EXC] Не удалось открыть CSV: {e}")

    def _show_examples(self) -> None:
        examples = read_examples_from_csv(OUTPUT_CSV, limit=3)
        self._render_examples_cards(self.examples_container, examples)

    def _render_examples_cards(self, parent: tk.Widget, examples: List[Dict[str, str]]) -> None:
        for w in parent.winfo_children():
            w.destroy()
        if not examples:
            ttk.Label(parent, text="Нет примеров. Сначала запустите пайплайн.").pack(anchor="w")
            return
        for ex in examples:
            card = ttk.Frame(parent, padding=12)
            card.pack(fill="x", expand=True, pady=6)
            card.configure(style="Card.TFrame")

            header = f"Клиент {ex['client_code']} → {ex['product']}"
            ttk.Label(card, text=header, font=("Segoe UI", 11, "bold")).pack(anchor="w")
            text_box = tk.Text(card, height=4, wrap="word")
            text_box.pack(fill="x", padx=(0, 0))
            text_box.insert("end", ex['push_notification'])
            text_box.configure(state="disabled")


def read_examples_from_csv(path: str, limit: int = 3) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append({
                "client_code": row.get("client_code", ""),
                "product": row.get("product", ""),
                "push_notification": row.get("push_notification", "")
            })
    return rows


if __name__ == "__main__":
    app = DemoApp()
    # Simple card style
    style = ttk.Style()
    style.configure("Card.TFrame", relief="groove", borderwidth=1)
    app.mainloop()

