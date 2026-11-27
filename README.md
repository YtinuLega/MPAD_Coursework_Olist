# Дослідження поведінки покупців в електронній комерції

Програмний проєкт до курсової роботи з дисципліни **«Мови програмування для аналізу даних»**  
Національний університет «Полтавська політехніка імені Юрія Кондратюка»  
Кафедра комп’ютерних та інформаційних технологій і систем, спеціальність 122 «Комп’ютерні науки»

**Виконав:** студент групи 403-ТН Павлюк Олександр Васильович  
**Керівник:** старший викладач кафедри КІТС Гайтан Олена Миколаївна

---

## Мета проєкту
Дослідити закономірності поведінки покупців у секторі електронної комерції на основі реальних транзакційних даних: динаміку замовлень і виручки у часі, структуру попиту за категоріями, платіжні звички (зокрема використання розстрочки), вплив тривалості та затримок доставки на оцінки клієнтів, а також частку повторних покупок.

## Джерело даних
Використано відкритий набір даних **Olist Brazilian E-Commerce Public Dataset** (період 2016–2018 рр., 99 441 замовлення), опублікований компанією Olist на GitHub та Kaggle:
- GitHub: `https://github.com/olist/work-at-olist-data/tree/master/datasets`
- Kaggle: `https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce`

Склад вихідних файлів у `data/raw/`:
1. `olist_orders_dataset.csv` - замовлення, статуси та часові мітки доставки;
2. `olist_order_items_dataset.csv` - позиції замовлень, ціни товарів і вартість доставки;
3. `olist_customers_dataset.csv` - покупці, унікальні ідентифікатори (`customer_unique_id`), міста й штати;
4. `olist_products_dataset.csv` - характеристики й категорії товарів;
5. `olist_order_payments_dataset.csv` - способи оплати, суми та кількість платежів у розстрочку;
6. `olist_order_reviews_dataset.csv` - оцінки замовлень покупцями (1-5 балів);
7. `product_category_name_translation.csv` - переклад назв категорій англійською мовою.

## Технології
- Python 3.11
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook
- Tableau Public (опублікований дашборд: `https://public.tableau.com/app/profile/oleksandr.pavliuk/viz/Olist_17955789686950/sheet3`)

## Структура проєкту
```text
MPAD_Coursework_Olist/
├── data/
│   ├── raw/                  # Вихідні CSV-таблиці Olist
│   └── processed/            # Очищена зведена вітрина замовлень (створюється під час запуску)
├── notebooks/
│   └── analysis.ipynb        # Основний ноутбук із повним циклом аналізу
├── src/
│   └── analysis_pipeline.py  # Скрипт відтворення розрахунків, графіків і експортів
├── results/
│   ├── figures/              # 7 підготовлених графіків (PNG)
│   └── exports/              # Агреговані CSV-таблиці та summary_stats.json
├── tableau/
│   ├── ecommerce_tableau_aggregated.xlsx  # Агрегований файл для Tableau Public
│   ├── tableau_public_overview.png        # Скриншот опублікованого дашборда
│   └── tableau_public_filtered.png        # Скриншот роботи інтерактивного фільтра
├── docs/
│   └── Курсова_робота_МПАД_Павлюк_403ТН.docx
├── requirements.txt
├── .gitignore
└── README.md
```

## Інструкція із запуску
1. Встановити залежності:
   ```bash
   pip install -r requirements.txt
   ```
2. Завантажити вихідні файли Olist у папку `data/raw/` (якщо вони виключені через `.gitignore`).
3. Запустити повний пайплайн розрахунків та генерації графіків:
   ```bash
   python src/analysis_pipeline.py
   ```
4. Або відкрити інтерактивний ноутбук:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```
