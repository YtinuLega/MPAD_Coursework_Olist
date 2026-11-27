# Курсова робота з дисципліни "Мови програмування для аналізу даних"
# Тема: Дослідження поведінки покупців в електронній комерції
# Автор: Павлюк Олександр Васильович, група 403-ТН

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Базові налаштування відображення
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8
sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
EXP_DIR = os.path.join(BASE_DIR, "results", "exports")
TAB_DIR = os.path.join(BASE_DIR, "tableau")

for d in [PROC_DIR, FIG_DIR, EXP_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)


def run_analysis():
    stats = {}

    # ==========================================================
    # 1. Завантаження вихідних таблиць (Olist Brazilian E-Commerce)
    # ==========================================================
    orders = pd.read_csv(os.path.join(RAW_DIR, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(RAW_DIR, "olist_order_items_dataset.csv"))
    customers = pd.read_csv(os.path.join(RAW_DIR, "olist_customers_dataset.csv"))
    products = pd.read_csv(os.path.join(RAW_DIR, "olist_products_dataset.csv"))
    payments = pd.read_csv(os.path.join(RAW_DIR, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(os.path.join(RAW_DIR, "olist_order_reviews_dataset.csv"))
    cat_trans = pd.read_csv(os.path.join(RAW_DIR, "product_category_name_translation.csv"))

    raw_tables = {
        "orders": orders,
        "order_items": items,
        "customers": customers,
        "products": products,
        "payments": payments,
        "reviews": reviews,
        "category_translation": cat_trans,
    }

    print("=== 1. Розміри вихідних таблиць ===")
    table_info = {}
    for name, tbl in raw_tables.items():
        n_rows, n_cols = tbl.shape
        n_dups = int(tbl.duplicated().sum())
        n_null_cells = int(tbl.isna().sum().sum())
        null_cols = {k: int(v) for k, v in tbl.isna().sum().items() if v > 0}
        table_info[name] = {
            "rows": n_rows,
            "cols": n_cols,
            "exact_duplicates": n_dups,
            "total_nulls": n_null_cells,
            "null_cols": null_cols,
        }
        print(f"{name:22s}: {n_rows:7d} рядків, {n_cols:2d} стовпців, дублікатів={n_dups}, пропусків={n_null_cells}")
    stats["raw_tables"] = table_info

    # Розподіл статусів усіх замовлень до фільтрації
    status_counts = orders["order_status"].value_counts()
    stats["order_statuses"] = {k: int(v) for k, v in status_counts.items()}
    print("\nСтатуси замовлень (усього):")
    print(status_counts)

    # ==========================================================
    # 2. Попереднє опрацювання та очищення даних
    # ==========================================================
    # Переводимо текстові дати у формат datetime
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    date_min = str(orders["order_purchase_timestamp"].min())
    date_max = str(orders["order_purchase_timestamp"].max())
    stats["raw_date_range"] = [date_min, date_max]

    # Перевіряємо пропуски в orders до фільтрації
    stats["orders_nulls_before"] = {k: int(v) for k, v in orders.isna().sum().items()}

    # Залишаємо доставлені замовлення з коректною датою доставки
    # (скасовані та недоступні замовлення не мають дати доставки за змістом)
    orders_deliv = orders[orders["order_status"] == "delivered"].copy()
    missing_deliv_date = int(orders_deliv["order_delivered_customer_date"].isna().sum())
    orders_clean = orders_deliv.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"]).copy()
    # Також перевіряємо, щоб дата доставки не була ранішою за дату покупки
    bad_dates = int((orders_clean["order_delivered_customer_date"] < orders_clean["order_purchase_timestamp"]).sum())
    orders_clean = orders_clean[orders_clean["order_delivered_customer_date"] >= orders_clean["order_purchase_timestamp"]].copy()

    stats["cleaning_orders"] = {
        "total_orders": int(len(orders)),
        "delivered_orders": int(len(orders_deliv)),
        "delivered_missing_date_dropped": missing_deliv_date,
        "negative_delivery_time_dropped": bad_dates,
        "orders_clean_count": int(len(orders_clean)),
    }

    # Доповнюємо переклад категорій для 2 португальських назв, яких немає у словнику
    extra_trans = pd.DataFrame({
        "product_category_name": [
            "pc_gamer",
            "portateis_cozinha_e_preparadores_de_alimentos",
        ],
        "product_category_name_english": [
            "pc_gamer",
            "portable_kitchen_food_preparers",
        ],
    })
    cat_trans_full = pd.concat([cat_trans, extra_trans], ignore_index=True).drop_duplicates(
        subset=["product_category_name"]
    )

    # Об'єднуємо товари з англійськими назвами категорій
    products_clean = products.merge(cat_trans_full, on="product_category_name", how="left")
    missing_cat_before = int(products_clean["product_category_name_english"].isna().sum())
    products_clean["category"] = products_clean["product_category_name_english"].fillna("unknown")

    # Заповнюємо поодинокі пропуски у вазі товару медіаною відповідної категорії (або загальною медіаною)
    missing_weight_before = int(products_clean["product_weight_g"].isna().sum())
    cat_weight_med = products_clean.groupby("category")["product_weight_g"].transform("median")
    products_clean["product_weight_g"] = products_clean["product_weight_g"].fillna(cat_weight_med)
    products_clean["product_weight_g"] = products_clean["product_weight_g"].fillna(products_clean["product_weight_g"].median())

    stats["cleaning_products"] = {
        "total_products": int(len(products)),
        "missing_category_filled_unknown": missing_cat_before,
        "missing_weight_filled_median": missing_weight_before,
    }

    # Дедуплікація відгуків (у частині замовлень клієнти лишали більше одного відгуку)
    reviews["review_creation_date"] = pd.to_datetime(reviews["review_creation_date"], errors="coerce")
    dup_review_orders = int(reviews.duplicated(subset=["order_id"]).sum())
    reviews_clean = (
        reviews.sort_values("review_creation_date")
        .drop_duplicates(subset=["order_id"], keep="last")
        [["order_id", "review_score"]]
    )
    stats["cleaning_reviews"] = {
        "raw_reviews": int(len(reviews)),
        "duplicate_order_reviews_removed": dup_review_orders,
        "clean_reviews": int(len(reviews_clean)),
    }

    # Перевірка цін і доставки в order_items
    neg_prices = int((items["price"] <= 0).sum())
    neg_freight = int((items["freight_value"] < 0).sum())
    stats["cleaning_items"] = {
        "raw_items": int(len(items)),
        "non_positive_price": neg_prices,
        "negative_freight": neg_freight,
    }

    # ==========================================================
    # 3. Об'єднання таблиць (merge) у єдину аналітичну вітрину
    # ==========================================================
    # 3.1 Поєднуємо позиції замовлень із товарами та категоріями
    items_prod = items.merge(
        products_clean[["product_id", "category", "product_weight_g"]],
        on="product_id",
        how="left",
    )

    # Агрегуємо позиції на рівень одного замовлення
    # Для основної категорії замовлення беремо товар з найбільшою ціною в кошику
    items_sorted = items_prod.sort_values(["order_id", "price"], ascending=[True, False])
    main_cat_per_order = items_sorted.drop_duplicates(subset=["order_id"])[["order_id", "category"]]

    items_agg = items_prod.groupby("order_id", as_index=False).agg(
        items_count=("order_item_id", "count"),
        unique_products=("product_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
    )
    items_agg = items_agg.merge(main_cat_per_order, on="order_id", how="left")

    # 3.2 Агрегуємо платежі на рівень замовлення (основний спосіб оплати - з найбільшою сумою)
    pay_sorted = payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
    main_pay = pay_sorted.drop_duplicates(subset=["order_id"])[["order_id", "payment_type"]]
    pay_agg = payments.groupby("order_id", as_index=False).agg(
        payment_total=("payment_value", "sum"),
        max_installments=("payment_installments", "max"),
        payment_parts=("payment_sequential", "count"),
    )
    pay_agg = pay_agg.merge(main_pay, on="order_id", how="left")

    # 3.3 Послідовні об'єднання (4 реляційних merge)
    df = orders_clean.merge(
        customers[["customer_id", "customer_unique_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
    )
    df = df.merge(items_agg, on="order_id", how="inner")
    df = df.merge(pay_agg, on="order_id", how="left")
    df = df.merge(reviews_clean, on="order_id", how="left")

    # У кількох замовлень можливий пропуск payment_type (1 замовлення в Olist)
    missing_pay_after_merge = int(df["payment_type"].isna().sum())
    df["payment_type"] = df["payment_type"].fillna("not_defined")
    df["max_installments"] = df["max_installments"].fillna(1).astype(int)

    stats["merged_dataset"] = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "unique_customers": int(df["customer_unique_id"].nunique()),
        "missing_payment_after_merge": missing_pay_after_merge,
        "missing_review_after_merge": int(df["review_score"].isna().sum()),
    }
    print(f"\n=== 2. Підсумкова вітрина після merge ===")
    print(f"Рядків: {len(df)}, стовпців: {df.shape[1]}, унікальних покупців: {df['customer_unique_id'].nunique()}")

    # ==========================================================
    # 4. Створення нових аналітичних ознак (Feature Engineering)
    # ==========================================================
    # 1) Повна вартість замовлення (товари + доставка)
    df["order_total"] = (df["total_price"] + df["total_freight"]).round(2)

    # 2) Частка вартості доставки у повному чеку (%)
    df["freight_share_pct"] = (df["total_freight"] / df["order_total"] * 100).round(2)

    # 3) Фактичний час доставки в днях
    df["delivery_days"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400.0
    ).round(2)

    # 4) Відхилення від прогнозованої дати доставки (додатне значення = запізнення)
    df["delivery_delay_days"] = (
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.total_seconds() / 86400.0
    ).round(2)
    df["is_delayed"] = np.where(df["delivery_delay_days"] > 0, "Затримка", "Вчасно")

    # 5) Часові ознаки покупки
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.dayofweek  # 0=Пн .. 6=Нд
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    # 6) Сегмент чеку (pd.cut)
    df["check_segment"] = pd.cut(
        df["order_total"],
        bins=[0, 60, 150, 350, np.inf],
        labels=["Малий (<60)", "Середній (60-150)", "Вище середнього (150-350)", "Великий (>350)"],
    )

    # 7) Ознака повторного покупця
    cust_order_cnt = df.groupby("customer_unique_id")["order_id"].transform("count")
    df["customer_orders_cnt"] = cust_order_cnt
    df["customer_type"] = np.where(df["customer_orders_cnt"] > 1, "Повторний", "Одноразовий")

    # Зберігаємо очищену вітрину у data/processed
    proc_path = os.path.join(PROC_DIR, "ecommerce_clean_orders.csv")
    df.to_csv(proc_path, index=False)

    # ==========================================================
    # 5. Статистичний та аналітичний аналіз
    # ==========================================================
    num_cols = ["order_total", "total_price", "total_freight", "freight_share_pct",
                "items_count", "delivery_days", "delivery_delay_days", "max_installments", "review_score"]
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    desc["iqr"] = desc["75%"] - desc["25%"]
    desc["skew"] = df[num_cols].skew()
    desc = desc[["count", "mean", "median", "std", "min", "25%", "75%", "max", "iqr", "skew"]].round(2)
    desc.to_csv(os.path.join(EXP_DIR, "descriptive_stats.csv"))
    stats["descriptive_stats"] = desc.to_dict(orient="index")

    print("\n=== 3. Описова статистика ===")
    print(desc[["mean", "median", "std", "min", "25%", "75%", "max"]])

    # Аналіз викидів за правилом IQR для order_total та delivery_days
    outliers_info = {}
    for col in ["order_total", "delivery_days", "freight_share_pct"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        cnt_out = int((df[col] > upper).sum())
        pct_out = round(cnt_out / len(df) * 100, 2)
        outliers_info[col] = {
            "q1": round(float(q1), 2),
            "q3": round(float(q3), 2),
            "iqr": round(float(iqr), 2),
            "upper_fence": round(float(upper), 2),
            "outliers_count": cnt_out,
            "outliers_pct": pct_out,
        }
    stats["outliers_iqr"] = outliers_info

    # Загальні KPI
    stats["kpi"] = {
        "total_revenue": round(float(df["order_total"].sum()), 2),
        "total_product_revenue": round(float(df["total_price"].sum()), 2),
        "total_freight_revenue": round(float(df["total_freight"].sum()), 2),
        "avg_check": round(float(df["order_total"].mean()), 2),
        "median_check": round(float(df["order_total"].median()), 2),
        "avg_delivery_days": round(float(df["delivery_days"].mean()), 2),
        "median_delivery_days": round(float(df["delivery_days"].median()), 2),
        "delayed_orders_cnt": int((df["is_delayed"] == "Затримка").sum()),
        "delayed_orders_pct": round(float((df["is_delayed"] == "Затримка").mean() * 100), 2),
        "avg_review_score": round(float(df["review_score"].mean()), 2),
        "avg_review_ontime": round(float(df.loc[df["is_delayed"] == "Вчасно", "review_score"].mean()), 2),
        "avg_review_delayed": round(float(df.loc[df["is_delayed"] == "Затримка", "review_score"].mean()), 2),
    }

    # --- Питання 1: Динаміка в часі (місячна + ковзне середнє) ---
    # Основний повний період для динаміки: січень 2017 - серпень 2018 (у 2016 є лише поодинокі тестові місяці)
    monthly = (
        df.set_index("order_purchase_timestamp")
        .resample("ME")
        .agg(
            orders_count=("order_id", "count"),
            revenue=("order_total", "sum"),
            avg_check=("order_total", "mean"),
            avg_delivery=("delivery_days", "mean"),
        )
        .reset_index()
    )
    monthly["month"] = monthly["order_purchase_timestamp"].dt.strftime("%Y-%m")
    monthly = monthly[monthly["orders_count"] >= 50].copy()  # відсікаємо неповні крайові місяці 2016 року
    monthly["orders_ma3"] = monthly["orders_count"].rolling(3, min_periods=1).mean().round(1)
    monthly["revenue_ma3"] = monthly["revenue"].rolling(3, min_periods=1).mean().round(2)
    monthly.to_csv(os.path.join(EXP_DIR, "monthly_dynamics.csv"), index=False)

    peak_row = monthly.loc[monthly["orders_count"].idxmax()]
    stats["monthly_peak"] = {
        "month": str(peak_row["month"]),
        "orders_count": int(peak_row["orders_count"]),
        "revenue": round(float(peak_row["revenue"]), 2),
        "avg_check": round(float(peak_row["avg_check"]), 2),
    }
    stats["monthly_table"] = monthly[["month", "orders_count", "revenue", "avg_check"]].round(2).to_dict(orient="records")

    # --- Питання 2: Топ-10 категорій (groupby + agg + sort_values + rank) ---
    cat_agg = (
        df.groupby("category", as_index=False)
        .agg(
            orders_count=("order_id", "count"),
            total_revenue=("order_total", "sum"),
            avg_check=("order_total", "mean"),
            median_check=("order_total", "median"),
            avg_review=("review_score", "mean"),
        )
        .sort_values("total_revenue", ascending=False)
    )
    cat_agg["revenue_share_pct"] = (cat_agg["total_revenue"] / cat_agg["total_revenue"].sum() * 100).round(2)
    cat_agg["revenue_rank"] = cat_agg["total_revenue"].rank(ascending=False, method="dense").astype(int)
    cat_agg = cat_agg.round(2)
    cat_agg.to_csv(os.path.join(EXP_DIR, "categories_summary.csv"), index=False)

    top10_cats = cat_agg.nlargest(10, "total_revenue")
    bottom5_cats = cat_agg[cat_agg["orders_count"] >= 20].nsmallest(5, "avg_check")
    stats["top10_categories"] = top10_cats.to_dict(orient="records")
    stats["top10_share_pct"] = round(float(top10_cats["revenue_share_pct"].sum()), 2)
    stats["total_categories"] = int(cat_agg["category"].nunique())
    stats["bottom5_avg_check_cats"] = bottom5_cats[["category", "orders_count", "avg_check"]].to_dict(orient="records")

    # --- Питання 3: Способи оплати та розстрочка ---
    pay_summary = (
        df.groupby("payment_type", as_index=False)
        .agg(
            orders_count=("order_id", "count"),
            total_revenue=("order_total", "sum"),
            avg_check=("order_total", "mean"),
            median_check=("order_total", "median"),
            avg_installments=("max_installments", "mean"),
        )
        .sort_values("orders_count", ascending=False)
    )
    pay_summary["orders_share_pct"] = (pay_summary["orders_count"] / len(df) * 100).round(2)
    pay_summary = pay_summary.round(2)
    pay_summary.to_csv(os.path.join(EXP_DIR, "payment_summary.csv"), index=False)
    stats["payment_summary"] = pay_summary.to_dict(orient="records")

    # Розстрочка для кредитних карток (2 замовлення мали значення 0 - відносимо до 1 платежу)
    cc_df = df[df["payment_type"] == "credit_card"].copy()
    cc_inst = (
        cc_df.groupby(pd.cut(cc_df["max_installments"].clip(lower=1), bins=[0, 1, 3, 6, 10, 30],
                             labels=["1 платіж", "2-3 платежі", "4-6 платежів", "7-10 платежів", "11+ платежів"]),
                      observed=False)
        .agg(orders_count=("order_id", "count"), avg_check=("order_total", "mean"), median_check=("order_total", "median"))
        .round(2)
    )
    stats["cc_installments_groups"] = cc_inst.reset_index().to_dict(orient="records")

    # --- Питання 4: Географія (штати), доставка та оцінки ---
    state_agg = (
        df.groupby("customer_state", as_index=False)
        .agg(
            orders_count=("order_id", "count"),
            total_revenue=("order_total", "sum"),
            avg_check=("order_total", "mean"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_freight_share=("freight_share_pct", "mean"),
            delayed_pct=("is_delayed", lambda s: (s == "Затримка").mean() * 100),
            avg_review=("review_score", "mean"),
        )
        .sort_values("orders_count", ascending=False)
        .round(2)
    )
    state_agg["orders_share_pct"] = (state_agg["orders_count"] / len(df) * 100).round(2)
    state_agg.to_csv(os.path.join(EXP_DIR, "states_summary.csv"), index=False)
    stats["top10_states"] = state_agg.head(10).to_dict(orient="records")
    stats["slowest5_states"] = state_agg.nlargest(5, "avg_delivery_days")[
        ["customer_state", "orders_count", "avg_delivery_days", "delayed_pct", "avg_review"]
    ].to_dict(orient="records")

    # Зведена таблиця 1: Топ-6 категорій x Спосіб оплати (pivot_table)
    top6_names = top10_cats["category"].head(6).tolist()
    pivot_cat_pay = pd.pivot_table(
        df[df["category"].isin(top6_names) & df["payment_type"].isin(["credit_card", "boleto", "voucher", "debit_card"])],
        index="category",
        columns="payment_type",
        values="order_total",
        aggfunc="mean",
        fill_value=0,
    ).round(2)
    pivot_cat_pay.to_csv(os.path.join(EXP_DIR, "pivot_category_payment.csv"))
    stats["pivot_cat_pay"] = pivot_cat_pay.to_dict(orient="index")

    # Зведена таблиця 2: Затримка доставки x Оцінка покупця (crosstab у %)
    ct_delay_review = pd.crosstab(df["is_delayed"], df["review_score"], normalize="index") * 100
    ct_delay_review = ct_delay_review.round(2)
    ct_delay_review.to_csv(os.path.join(EXP_DIR, "crosstab_delay_review.csv"))
    stats["crosstab_delay_review_pct"] = ct_delay_review.to_dict(orient="index")

    # --- Питання 5: Повторні покупці та сегменти чеку ---
    cust_lvl = df.groupby("customer_unique_id", as_index=False).agg(
        orders_cnt=("order_id", "count"),
        total_spent=("order_total", "sum"),
        avg_order=("order_total", "mean"),
    )
    cust_lvl["group"] = np.where(cust_lvl["orders_cnt"] > 1, "Повторні (2+ замовлень)", "Одноразові (1 замовлення)")
    repeat_summary = cust_lvl.groupby("group", as_index=False).agg(
        customers_cnt=("customer_unique_id", "count"),
        total_orders=("orders_cnt", "sum"),
        total_spent=("total_spent", "sum"),
        avg_order=("avg_order", "mean"),
        avg_ltv=("total_spent", "mean"),
    ).round(2)
    repeat_summary["customers_pct"] = (repeat_summary["customers_cnt"] / len(cust_lvl) * 100).round(2)
    repeat_summary["revenue_pct"] = (repeat_summary["total_spent"] / repeat_summary["total_spent"].sum() * 100).round(2)
    repeat_summary.to_csv(os.path.join(EXP_DIR, "repeat_customers.csv"), index=False)
    stats["repeat_customers"] = repeat_summary.to_dict(orient="records")

    seg_summary = df.groupby("check_segment", observed=False, as_index=False).agg(
        orders_count=("order_id", "count"),
        total_revenue=("order_total", "sum"),
        avg_freight_share=("freight_share_pct", "mean"),
        avg_installments=("max_installments", "mean"),
        avg_review=("review_score", "mean"),
    ).round(2)
    seg_summary["orders_pct"] = (seg_summary["orders_count"] / len(df) * 100).round(2)
    seg_summary["revenue_pct"] = (seg_summary["total_revenue"] / seg_summary["total_revenue"].sum() * 100).round(2)
    seg_summary.to_csv(os.path.join(EXP_DIR, "check_segments.csv"), index=False)
    stats["check_segments"] = seg_summary.to_dict(orient="records")

    # Кореляційна матриця (Пірсон і Спірмен)
    corr_cols = [
        "order_total",
        "total_freight",
        "freight_share_pct",
        "items_count",
        "total_weight_g",
        "max_installments",
        "delivery_days",
        "delivery_delay_days",
        "review_score",
    ]
    corr_pearson = df[corr_cols].corr(method="pearson").round(3)
    corr_spearman = df[corr_cols].corr(method="spearman").round(3)
    corr_pearson.to_csv(os.path.join(EXP_DIR, "corr_pearson.csv"))
    corr_spearman.to_csv(os.path.join(EXP_DIR, "corr_spearman.csv"))
    stats["key_correlations_pearson"] = {
        "order_total_vs_installments": float(corr_pearson.loc["order_total", "max_installments"]),
        "order_total_vs_freight": float(corr_pearson.loc["order_total", "total_freight"]),
        "weight_vs_freight": float(corr_pearson.loc["total_weight_g", "total_freight"]),
        "delivery_days_vs_review": float(corr_pearson.loc["delivery_days", "review_score"]),
        "delay_days_vs_review": float(corr_pearson.loc["delivery_delay_days", "review_score"]),
        "order_total_vs_review": float(corr_pearson.loc["order_total", "review_score"]),
    }
    stats["key_correlations_spearman"] = {
        "order_total_vs_installments": float(corr_spearman.loc["order_total", "max_installments"]),
        "delivery_days_vs_review": float(corr_spearman.loc["delivery_days", "review_score"]),
        "delay_days_vs_review": float(corr_spearman.loc["delivery_delay_days", "review_score"]),
    }

    # Активність за днями тижня та годинами
    wd_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
    wd_counts = df["purchase_weekday"].value_counts().sort_index()
    stats["weekday_orders"] = {wd_names[i]: int(wd_counts.get(i, 0)) for i in range(7)}
    hr_counts = df["purchase_hour"].value_counts().sort_index()
    stats["peak_hour"] = int(hr_counts.idxmax())
    stats["peak_hour_orders"] = int(hr_counts.max())
    stats["quietest_hour"] = int(hr_counts.idxmin())

    # ==========================================================
    # 6. Побудова 7 змістовних графіків (Matplotlib + Seaborn)
    # ==========================================================

    # Графік 1: Динаміка замовлень та виручки по місяцях (лінійний + ковзне середнє)
    fig, ax1 = plt.subplots(figsize=(10.5, 5.2))
    x_pos = np.arange(len(monthly))
    ax1.bar(x_pos, monthly["orders_count"], color="#5B8FF9", alpha=0.65, label="Кількість замовлень")
    ax1.plot(x_pos, monthly["orders_ma3"], color="#1D39C4", marker="o", linewidth=2.2, label="Ковзне середнє (3 міс.)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(monthly["month"], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Кількість доставлених замовлень, шт.", fontsize=10, color="#1D39C4")
    ax1.set_xlabel("Місяць покупки", fontsize=10)

    ax2 = ax1.twinx()
    ax2.plot(x_pos, monthly["revenue"] / 1000.0, color="#E8684A", linestyle="--", marker="s",
             linewidth=1.8, markersize=4, label="Загальний обіг, тис. BRL")
    ax2.set_ylabel("Обіг, тис. BRL", fontsize=10, color="#C93B2B")
    ax2.grid(False)

    ax1.set_title("Динаміка кількості замовлень та обігу інтернет-майданчика (2017–2018)", fontsize=12, fontweight="bold", pad=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_monthly_dynamics.png"), dpi=220)
    plt.close()

    # Графік 2: Топ-10 категорій товарів за обігом (горизонтальна стовпчикова діаграма)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    top10_plot = top10_cats.sort_values("total_revenue", ascending=True)
    bars = ax.barh(top10_plot["category"], top10_plot["total_revenue"] / 1e6, color=sns.color_palette("Blues_d", 10))
    for bar, (_, row) in zip(bars, top10_plot.iterrows()):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{row['total_revenue']/1e6:.2f} млн ({row['revenue_share_pct']:.1f}%)",
                va="center", fontsize=8.5)
    ax.set_xlim(0, top10_plot["total_revenue"].max() / 1e6 * 1.25)
    ax.set_xlabel("Сумарний обіг, млн BRL", fontsize=10)
    ax.set_ylabel("Категорія товару", fontsize=10)
    ax.set_title("Топ-10 категорій товарів за обсягом продажів", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_top_categories.png"), dpi=220)
    plt.close()

    # Графік 3: Розподіл суми замовлення (гістограма + KDE для 98% замовлень та лог-розподіл)
    p98 = df["order_total"].quantile(0.98)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sns.histplot(df.loc[df["order_total"] <= p98, "order_total"], bins=45, kde=True, color="#2E86AB", ax=axes[0])
    axes[0].axvline(df["order_total"].mean(), color="#D9534F", linestyle="--", linewidth=1.6,
                    label=f"Середнє: {df['order_total'].mean():.1f} BRL")
    axes[0].axvline(df["order_total"].median(), color="#2CA02C", linestyle="-", linewidth=1.6,
                    label=f"Медіана: {df['order_total'].median():.1f} BRL")
    axes[0].set_title("Розподіл суми замовлення (до 98-го перцентиля)", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Сума замовлення, BRL")
    axes[0].set_ylabel("Кількість замовлень")
    axes[0].legend(fontsize=8.5)

    sns.histplot(np.log10(df["order_total"].clip(lower=1)), bins=40, kde=True, color="#6C5B7B", ax=axes[1])
    axes[1].set_title("Логарифмічний розподіл суми замовлення (log10)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("log10(Сума замовлення, BRL)")
    axes[1].set_ylabel("Кількість замовлень")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_order_value_dist.png"), dpi=220)
    plt.close()

    # Графік 4: Boxplot часу доставки залежно від оцінки покупця (1-5 зірок)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    plot_rev = df.dropna(subset=["review_score"]).copy()
    plot_rev["review_score"] = plot_rev["review_score"].astype(int)
    sns.boxplot(
        data=plot_rev,
        x="review_score",
        y="delivery_days",
        hue="review_score",
        palette="RdYlGn",
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Залежність тривалості доставки від оцінки покупця (без екстремальних викидів)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Оцінка замовлення покупцем (бали від 1 до 5)", fontsize=10)
    ax.set_ylabel("Фактичний час доставки, днів", fontsize=10)
    # Підписуємо медіани над коробками з контрастним фоном
    meds = plot_rev.groupby("review_score")["delivery_days"].median()
    for i, score in enumerate(sorted(meds.index)):
        ax.text(
            i, meds[score] + 1.2, f"Мед.: {meds[score]:.1f} дн.",
            ha="center", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.88),
        )
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_delivery_review_boxplot.png"), dpi=220)
    plt.close()

    # Графік 5: Способи оплати та залежність кількості платежів у розстрочку від суми замовлення
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    pay_plot = pay_summary[pay_summary["payment_type"] != "not_defined"].copy()
    pay_labels = {
        "credit_card": "Кредитна картка",
        "boleto": "Boleto (рахунок)",
        "voucher": "Ваучер",
        "debit_card": "Дебетова картка",
    }
    pay_plot["label"] = pay_plot["payment_type"].map(pay_labels)
    sns.barplot(data=pay_plot, x="label", y="orders_share_pct", hue="label", palette="Set2", legend=False, ax=axes[0])
    for i, v in enumerate(pay_plot["orders_share_pct"]):
        axes[0].text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].set_ylim(0, 86)
    axes[0].set_title("Структура замовлень за способом оплати", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Спосіб оплати")
    axes[0].set_ylabel("Частка замовлень, %")
    axes[0].tick_params(axis="x", rotation=15)

    # Діаграма розсіювання (підвибірка 4000 записів для читабельності) + лінія тренду
    cc_sample = cc_df[cc_df["order_total"] <= 2000].sample(4000, random_state=42)
    sns.regplot(
        data=cc_sample,
        x="order_total",
        y="max_installments",
        scatter_kws={"alpha": 0.25, "s": 14, "color": "#3B7DDD"},
        line_kws={"color": "#D9534F", "linewidth": 2},
        ax=axes[1],
    )
    axes[1].set_title("Сума чеку та кількість платежів (кредитні картки)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Сума замовлення, BRL")
    axes[1].set_ylabel("Кількість платежів у розстрочку")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_payment_installments.png"), dpi=220)
    plt.close()

    # Графік 6: Теплова карта кореляційної матриці (Seaborn heatmap)
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    ua_labels = [
        "Сума замовл.",
        "Доставка (BRL)",
        "Частка дост. (%)",
        "К-сть товарів",
        "Вага (г)",
        "Розстрочка",
        "Дні доставки",
        "Затримка (дн.)",
        "Оцінка (1–5)",
    ]
    corr_disp = corr_pearson.copy()
    corr_disp.index = ua_labels
    corr_disp.columns = ua_labels
    mask = np.triu(np.ones_like(corr_disp, dtype=bool), k=1)
    sns.heatmap(
        corr_disp,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-0.8,
        vmax=0.8,
        square=True,
        linewidths=0.6,
        ax=ax,
    )
    ax.set_title("Кореляційна матриця (коефіцієнт Пірсона) основних показників", fontsize=11.5, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_correlation_heatmap.png"), dpi=220)
    plt.close()

    # Графік 7: Теплова карта активності покупців (день тижня x година доби)
    wd_hr = pd.crosstab(df["purchase_weekday"], df["purchase_hour"])
    wd_hr.index = wd_names
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    sns.heatmap(wd_hr, cmap="YlGnBu", linewidths=0.3, ax=ax, cbar_kws={"label": "Кількість замовлень"})
    ax.set_title("Активність оформлення замовлень за днем тижня та годиною доби", fontsize=11.5, fontweight="bold", pad=10)
    ax.set_xlabel("Година доби (0–23)")
    ax.set_ylabel("День тижня")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_weekday_hour_heatmap.png"), dpi=220)
    plt.close()

    # ==========================================================
    # 7. Експорт підготовлених даних для Tableau Public
    # ==========================================================
    tableau_cols = [
        "order_id",
        "order_purchase_timestamp",
        "purchase_year",
        "purchase_month",
        "customer_state",
        "customer_city",
        "category",
        "payment_type",
        "check_segment",
        "customer_type",
        "is_delayed",
        "items_count",
        "total_price",
        "total_freight",
        "order_total",
        "freight_share_pct",
        "max_installments",
        "delivery_days",
        "delivery_delay_days",
        "review_score",
    ]
    tableau_csv = os.path.join(TAB_DIR, "ecommerce_tableau_orders.csv")
    df[tableau_cols].to_csv(tableau_csv, index=False)

    # Агрегований Excel-файл (малий обсяг, швидко вантажиться у Tableau Public Web Authoring)
    tab_agg = (
        df.groupby(["purchase_month", "customer_state", "category", "payment_type", "is_delayed"], as_index=False)
        .agg(
            orders_count=("order_id", "count"),
            total_revenue=("order_total", "sum"),
            total_freight=("total_freight", "sum"),
            avg_delivery_days=("delivery_days", "mean"),
            avg_review_score=("review_score", "mean"),
        )
        .round(2)
    )
    tableau_xlsx = os.path.join(TAB_DIR, "ecommerce_tableau_aggregated.xlsx")
    tab_agg.to_excel(tableau_xlsx, index=False, sheet_name="EcommerceData")

    # Зберігаємо всі обчислені метрики у JSON для звіту
    stats_path = os.path.join(EXP_DIR, "summary_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== Усі розрахунки, графіки (7 шт.) та експорти для Tableau успішно створено! ===")
    print("Статистика збережена у:", stats_path)


if __name__ == "__main__":
    run_analysis()
