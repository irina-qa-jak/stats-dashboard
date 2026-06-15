from flask import Flask
import os
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import statsmodels.api as sm
from flask import render_template, request
from scipy import stats
import pandas as pd
import plotly.express as px
from scipy import stats
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import fisher_exact
from sklearn.linear_model import LinearRegression as SkLinearRegression
from scipy.stats import levene
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.linear_model import LinearRegression
from flask import jsonify
from scipy.stats import pearsonr
from scipy.stats import kendalltau
from scipy.stats import shapiro
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import mannwhitneyu
from flask import render_template, request, flash
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression  # модель
from sklearn.metrics import accuracy_score, roc_curve  # метрики



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.secret_key = "your_secret_key"

def parse_input_data(data_str):
    """Преобразует строку или список в список float."""
    if not data_str:
        return []
    try:
        if isinstance(data_str, list):
            return [float(x) for x in data_str if str(x).strip()]
        return [float(x.strip()) for x in str(data_str).replace(',', ' ').split() if x.strip()]
    except (ValueError, AttributeError):
        return []



def create_anova_plots(df, labels):
    group_stats = df.groupby('Group')['Value'].agg(['mean', 'std', 'count'])

    # Boxplot
    boxplot = go.Figure()
    for i, label in enumerate(labels):
        boxplot.add_trace(go.Box(
            y=df[df['Group'] == label]['Value'],
            name=label,
            boxpoints='all',
            jitter=0.3,
            marker_color=f'hsl({i * 40}, 50%, 50%)'
        ))
    boxplot_html = boxplot.to_html(full_html=False, include_plotlyjs='cdn')

    # Summary plot
    summary_fig = go.Figure()
    for i, label in enumerate(labels):
        summary_fig.add_trace(go.Box(
            y=df[df['Group'] == label]['Value'],
            name=label,
            boxpoints='outliers',
            marker_color=f'hsl({i * 40}, 50%, 50%)'
        ))
    summary_fig.add_trace(go.Scatter(
        x=labels,
        y=group_stats['mean'],
        error_y=dict(type='data', array=group_stats['std'], visible=True),
        mode='markers+lines',
        name='Mean ± SD',
        marker=dict(size=10, color='black')
    ))
    summary_plot_html = summary_fig.to_html(full_html=False, include_plotlyjs=False)

    # Means with 95% CI
    means_fig = go.Figure()
    ci95 = 1.96 * group_stats['std'] / np.sqrt(group_stats['count'])
    means_fig.add_trace(go.Bar(
        x=labels,
        y=group_stats['mean'],
        error_y=dict(type='data', array=ci95, visible=True),
        marker_color=[f'hsl({i * 40}, 50%, 50%)' for i in range(len(labels))]
    ))
    means_fig.update_layout(title='Means with 95% Confidence Intervals')
    means_plot_html = means_fig.to_html(full_html=False, include_plotlyjs=False)

    # Distribution plot
    dist_fig = go.Figure()
    for i, label in enumerate(labels):
        dist_fig.add_trace(go.Histogram(
            x=df[df['Group'] == label]['Value'],
            name=label,
            opacity=0.7,
            marker_color=f'hsl({i * 40}, 50%, 50%)'
        ))
    dist_fig.update_layout(title='Distribution by Group', barmode='overlay')
    distribution_plot_html = dist_fig.to_html(full_html=False, include_plotlyjs=False)

    return {
        'boxplot': boxplot_html,
        'summary_plot': summary_plot_html,
        'means_plot': means_plot_html,
        'distribution_plot': distribution_plot_html
    }

# ====== МАРШРУТЫ ======

# === Главная страница ===
@app.route("/", methods=['GET'])
def index_test():
    return render_template("index.html", active_page="home")


# === О проекте ===
@app.route("/about")
def about_test():
    return render_template(
        "about.html",
        active_page="about"
    )



# === ANOVA ===
@app.route("/anova", methods=["GET", "POST"])
def anova_test():
    from scipy import stats
    import pandas as pd
    import plotly.express as px

    stat = p_value = conclusion = None
    plot_box = plot_bar = plot_violin = plot_hist = None

    if request.method == "POST":
        try:
            groups = []
            i = 1
            while f"group{i}" in request.form:
                raw = request.form.get(f"group{i}", "").strip()
                if raw:
                    values = [float(x) for x in raw.replace(",", " ").split()]
                    groups.append(values)
                i += 1

            if len(groups) < 2:
                flash("Введите хотя бы две группы", "error")
            else:
                # ANOVA
                stat, p_value = stats.f_oneway(*groups)
                conclusion = (
                    "Различия статистически значимы (p < 0.05)"
                    if p_value < 0.05
                    else "Различия статистически незначимы (p ≥ 0.05)"
                )

                # DataFrame для графиков
                labels = [f"Группа {idx+1}" for idx, vals in enumerate(groups) for _ in vals]
                data = [val for vals in groups for val in vals]
                df = pd.DataFrame({"Value": data, "Group": labels})

                # 1. Boxplot
                fig_box = px.box(df, x="Group", y="Value", points="all", title="Boxplot по группам")
                plot_box = fig_box.to_html(full_html=False)

                # 2. Barplot (средние)
                df_mean = df.groupby("Group", as_index=False)["Value"].mean()
                fig_bar = px.bar(df_mean, x="Group", y="Value", title="Средние значения по группам")
                plot_bar = fig_bar.to_html(full_html=False)

                # 3. Violin plot
                fig_violin = px.violin(df, x="Group", y="Value", box=True, points="all", title="Violin Plot по группам")
                plot_violin = fig_violin.to_html(full_html=False)

                # 4. Histogram
                fig_hist = px.histogram(df, x="Value", color="Group", barmode="overlay", nbins=10, title="Гистограмма распределений")
                plot_hist = fig_hist.to_html(full_html=False)

        except ValueError:
            flash("Ошибка в данных. Проверьте ввод.", "error")

    return render_template(
        "anova.html",
        stat=stat,
        p_value=p_value,
        conclusion=conclusion,
        plot_box=plot_box,
        plot_bar=plot_bar,
        plot_violin=plot_violin,
        plot_hist=plot_hist,
        active_page="anova"
    )



# === t-test ===
@app.route("/ttest", methods=['GET', 'POST'])
def ttest_test():
    from scipy import stats
    import pandas as pd
    import plotly.express as px

    t_value = p_value = conclusion = None
    plot_box = plot_bar = plot_violin = plot_hist = None

    if request.method == "POST":
        try:
            # Получаем данные
            data1 = request.form.get("group1", "").strip()
            data2 = request.form.get("group2", "").strip()

            if not data1 or not data2:
                flash("Введите данные для обеих групп", "error")
            else:
                group1 = [float(x) for x in data1.replace(",", " ").split()]
                group2 = [float(x) for x in data2.replace(",", " ").split()]

                # Проверка на пустоту
                if not group1 or not group2:
                    flash("Введите данные для обеих групп", "error")
                else:
                    # Сам тест
                    t_value, p_value = stats.ttest_ind(group1, group2, equal_var=False)
                    conclusion = (
                        "Различия статистически значимы (p < 0.05)"
                        if p_value < 0.05
                        else "Различия статистически незначимы (p ≥ 0.05)"
                    )

                    # DataFrame
                    df = pd.DataFrame({
                        "Value": group1 + group2,
                        "Group": ["Группа 1"] * len(group1) + ["Группа 2"] * len(group2)
                    })

                    # 1. Boxplot
                    fig_box = px.box(df, x="Group", y="Value", points="all", title="Boxplot по группам")
                    plot_box = fig_box.to_html(full_html=False)

                    # 2. Barplot
                    df_mean = df.groupby("Group", as_index=False)["Value"].mean()
                    fig_bar = px.bar(df_mean, x="Group", y="Value", title="Средние значения по группам")
                    plot_bar = fig_bar.to_html(full_html=False)

                    # 3. Violin Plot
                    fig_violin = px.violin(df, x="Group", y="Value", box=True, points="all", title="Violin Plot")
                    plot_violin = fig_violin.to_html(full_html=False)

                    # 4. Histogram
                    fig_hist = px.histogram(df, x="Value", color="Group", barmode="overlay", nbins=10, title="Гистограмма")
                    plot_hist = fig_hist.to_html(full_html=False)

        except ValueError:
            flash("Ошибка в данных. Проверьте ввод.", "error")

    return render_template(
        "ttest.html",
        t_value=t_value,
        p_value=p_value,
        conclusion=conclusion,
        plot_box=plot_box,
        plot_bar=plot_bar,
        plot_violin=plot_violin,
        plot_hist=plot_hist,
        active_page="ttest"
    )



# === Pearson ===
@app.route('/pearson', methods=['GET', 'POST'])
def pearson_test():
    correlation = None
    p_value = None
    conclusion = None
    plots = {}
    error = None

    def _parse_input(s):
        if not s:
            return []
        try:
            return [float(x) for x in s.replace(',', ' ').split() if x.strip() != ""]
        except:
            return []

    if request.method == 'POST':
        try:
            x_data = _parse_input(request.form.get('x_data', ''))
            y_data = _parse_input(request.form.get('y_data', ''))

            if not x_data or not y_data:
                error = "Введите данные для X и Y."
            elif len(x_data) != len(y_data):
                error = "Количество значений в X и Y должно совпадать."
            elif len(x_data) < 2:
                error = "Нужно минимум по 2 значения."
            else:
                # тест Пирсона
                correlation, p_value = pearsonr(x_data, y_data)
                conclusion = (
                    "Есть статистически значимая корреляция." if p_value < 0.05
                    else "Статистически значимой корреляции не обнаружено."
                )

                # DataFrame
                df = pd.DataFrame({"X": x_data, "Y": y_data})

                # 1. Scatter с линией тренда
                fig1 = px.scatter(df, x="X", y="Y", trendline="ols",
                                  title="Scatter Plot с линией регрессии")
                plots['scatter'] = fig1.to_html(full_html=False, include_plotlyjs='cdn')

                # 2. Scatter с плотностью точек
                fig2 = px.density_heatmap(df, x="X", y="Y", nbinsx=20, nbinsy=20,
                                          title="Плотность распределения точек")
                plots['density'] = fig2.to_html(full_html=False, include_plotlyjs='cdn')

                # 3. Гистограммы X и Y
                fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Гистограмма X", "Гистограмма Y"))
                fig3.add_trace(go.Histogram(x=df["X"], nbinsx=10, name="X"), row=1, col=1)
                fig3.add_trace(go.Histogram(x=df["Y"], nbinsx=10, name="Y"), row=1, col=2)
                fig3.update_layout(title="Гистограммы переменных", showlegend=False)
                plots['histograms'] = fig3.to_html(full_html=False, include_plotlyjs='cdn')

                # 4. Тепловая карта корреляции
                corr_matrix = df.corr()
                fig4 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r',
                                 title="Матрица корреляций")
                plots['heatmap'] = fig4.to_html(full_html=False, include_plotlyjs='cdn')

        except Exception as e:
            error = str(e)

    return render_template(
        "pearson.html",
        correlation=correlation,
        p_value=p_value,
        conclusion=conclusion,
        plots=plots,
        error=error,
        active_page="pearson"
    )




# === Spearman ===
@app.route("/spearman", methods=["GET", "POST"])
def spearman_test():
    correlation = None
    p_value = None
    plots = {}
    error = None

    if request.method == "POST":
        try:
            # Получаем и парсим данные
            x_data_raw = request.form.get("x_data", "")
            y_data_raw = request.form.get("y_data", "")

            x_data = parse_input_data(x_data_raw)
            y_data = parse_input_data(y_data_raw)

            if not x_data or not y_data:
                error = "Введите данные для обеих групп"
            elif len(x_data) != len(y_data):
                error = "X и Y должны содержать одинаковое количество значений"
            else:
                # Считаем коэффициент Спирмена
                correlation, p_value = stats.spearmanr(x_data, y_data)

                # DataFrame
                df = pd.DataFrame({"X": x_data, "Y": y_data})

                # 4 графика
                plots["scatter"] = px.scatter(df, x="X", y="Y", trendline="ols",
                                              title="Диаграмма рассеяния").to_html(full_html=False)
                plots["hist_x"] = px.histogram(df, x="X", nbins=10,
                                               title="Гистограмма X").to_html(full_html=False)
                plots["hist_y"] = px.histogram(df, x="Y", nbins=10,
                                               title="Гистограмма Y").to_html(full_html=False)
                plots["box"] = px.box(df, y=["X", "Y"],
                                      title="Boxplot X и Y").to_html(full_html=False)

        except Exception as e:
            error = f"Ошибка обработки данных: {str(e)}"

    return render_template(
        "spearman.html",
        active_page="spearman",
        correlation=correlation,
        p_value=p_value,
        plots=plots,
        error=error
    )


# === Kendall ===
@app.route("/kendall", methods=["GET", "POST"])
def kendall_test():
    correlation = None
    p_value = None
    conclusion = None
    plots = {}
    error = None

    if request.method == "POST":
        try:
            # Парсим данные
            def parse_list(s):
                return [float(x) for x in s.replace(",", " ").split() if x.strip()]

            x_data = parse_list(request.form.get("x_data", ""))
            y_data = parse_list(request.form.get("y_data", ""))

            if not x_data or not y_data:
                error = "Введите данные для обеих переменных."
            elif len(x_data) != len(y_data):
                error = "Длины списков X и Y должны совпадать."
            else:
                # Вычисляем коэффициент Kendall
                correlation, p_value = stats.kendalltau(x_data, y_data)
                conclusion = (
                    "Есть статистически значимая связь (p < 0.05)."
                    if p_value < 0.05
                    else "Статистически значимой связи не выявлено."
                )

                # Готовим DataFrame для графиков
                df = pd.DataFrame({"X": x_data, "Y": y_data})

                # Scatter Plot
                plots["scatter"] = px.scatter(
                    df, x="X", y="Y", trendline="ols", title="Scatter Plot с линией регрессии"
                ).to_html(full_html=False, include_plotlyjs="cdn")

                # Гистограммы
                fig_hist = make_subplots(rows=1, cols=2, subplot_titles=("Гистограмма X", "Гистограмма Y"))
                fig_hist.add_trace(go.Histogram(x=df["X"], name="X"), row=1, col=1)
                fig_hist.add_trace(go.Histogram(x=df["Y"], name="Y"), row=1, col=2)
                fig_hist.update_layout(title="Гистограммы X и Y")
                plots["histograms"] = fig_hist.to_html(full_html=False, include_plotlyjs="cdn")

                # Violin Plot
                df_melted = df.melt(var_name="Variable", value_name="Value")
                plots["violin"] = px.violin(
                    df_melted, x="Variable", y="Value", box=True, points="all", title="Violin Plot"
                ).to_html(full_html=False, include_plotlyjs="cdn")

                # Матрица корреляций
                corr_matrix = df.corr(method="kendall")
                plots["heatmap"] = px.imshow(
                    corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", title="Матрица корреляций"
                ).to_html(full_html=False, include_plotlyjs="cdn")

        except Exception as e:
            error = f"Ошибка: {str(e)}"

    return render_template(
        "kendall.html",
        correlation=correlation,
        p_value=p_value,
        conclusion=conclusion,
        plots=plots,
        error=error,
        active_page="kendall"
    )



# === IQR ===
@app.route("/iqr", methods=["GET", "POST"])
def iqr_test():
    iqr_value = None
    q1 = None
    q3 = None
    plots = {}
    error = None

    if request.method == "POST":
        try:
            data_str = request.form.get("data", "")
            data = [float(x.strip()) for x in data_str.split(",") if x.strip()]
            if len(data) < 2:
                raise ValueError("Нужно хотя бы 2 числа")

            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr_value = q3 - q1

            # Boxplot
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(y=data, boxmean=True))
            fig_box.update_layout(title="Boxplot", template="plotly_white")
            plots["boxplot"] = fig_box.to_html(full_html=False)

            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=data, nbinsx=10))
            fig_hist.update_layout(title="Гистограмма", template="plotly_white")
            plots["histogram"] = fig_hist.to_html(full_html=False)

        except Exception as e:
            error = str(e)

    return render_template(
        "iqr.html",
        active_page="iqr",
        iqr_value=iqr_value,
        q1=q1,
        q3=q3,
        plots=plots,
        error=error
    )





# === Shapiro–Wilk ===
@app.route('/shapiro', methods=['GET', 'POST'])
def shapiro_test():
    stat = None
    p_value = None
    conclusion = None
    plots = {}

    if request.method == 'POST':
        try:
            raw_data = request.form.get('data', '').strip()
            if not raw_data:
                flash("Введите данные", "error")
                return render_template("shapiro.html", stat=stat, p_value=p_value, conclusion=conclusion, plots=plots, active_page="shapiro")

            # Преобразуем строку в список чисел
            data = [float(x) for x in raw_data.replace(',', ' ').split()]

            # Тест Шапиро-Уилка
            stat, p_value = stats.shapiro(data)
            conclusion = (
                "Данные распределены нормально (p ≥ 0.05)"
                if p_value >= 0.05
                else "Данные не распределены нормально (p < 0.05)"
            )

            df = pd.DataFrame({"Value": data})

            # 1. Гистограмма
            fig1 = px.histogram(df, x="Value", nbins=10, title="Гистограмма")
            plots["histogram"] = fig1.to_html(full_html=False)

            # 2. Boxplot
            fig2 = px.box(df, y="Value", points="all", title="Boxplot")
            plots["boxplot"] = fig2.to_html(full_html=False)

            # 3. QQ-Plot
            import statsmodels.api as sm
            import matplotlib.pyplot as plt
            import io, base64

            fig_qq = plt.figure()
            sm.qqplot(np.array(data), line='s', ax=plt.gca())
            plt.title("QQ-Plot")
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plots["qqplot"] = f'<img src="data:image/png;base64,{image_base64}"/>'
            plt.close(fig_qq)

            # 4. Kernel Density Estimation (KDE)
            fig4 = px.histogram(df, x="Value", nbins=10, marginal="violin", title="KDE + Гистограмма", histnorm='probability density')
            plots["kde"] = fig4.to_html(full_html=False)

        except ValueError:
            flash("Ошибка: введите корректные числа, разделённые пробелами или запятыми", "error")

    return render_template("shapiro.html",
                           stat=stat,
                           p_value=p_value,
                           conclusion=conclusion,
                           plots=plots,
                           active_page="shapiro")




# === Tukey HSD ===
@app.route("/tukey", methods=["GET", "POST"])
def tukey_test():
    try:
        if request.method == "POST":
            groups_data = request.form.getlist("group[]")
            data_list = []
            labels_list = []

            # Парсим все группы
            for idx, g in enumerate(groups_data, start=1):
                values = parse_input_data(g)
                if values:
                    data_list.extend(values)
                    labels_list.extend([f"Группа {idx}"] * len(values))

            # Проверка на минимум 2 группы
            if len(set(labels_list)) < 2:
                return render_template("tukey.html", error="Нужно минимум 2 группы.", active_page="tukey")

            # DataFrame для анализа
            df = pd.DataFrame({
                "Value": data_list,
                "Group": labels_list
            })

            # Выполняем ANOVA
            model = ols('Value ~ C(Group)', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)

            # Тест Тьюки
            tukey = pairwise_tukeyhsd(endog=df["Value"], groups=df["Group"], alpha=0.05)
            tukey_result = tukey.summary()

            # Построение графиков
            plots = {}

            # 1. Boxplot
            fig_box = px.box(df, x="Group", y="Value", points="all", title="Boxplot по группам")
            plots["boxplot"] = fig_box.to_html(full_html=False, include_plotlyjs='cdn')

            # 2. Violin plot
            fig_violin = px.violin(df, x="Group", y="Value", box=True, points="all", title="Violin Plot")
            plots["violin"] = fig_violin.to_html(full_html=False, include_plotlyjs='cdn')

            # 3. Средние значения с доверительными интервалами
            fig_mean = px.bar(
                df.groupby("Group", as_index=False)["Value"].mean(),
                x="Group", y="Value",
                title="Средние значения по группам"
            )
            plots["means"] = fig_mean.to_html(full_html=False, include_plotlyjs='cdn')

            # 4. Результаты Tukey
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            fig_tukey = px.scatter(tukey_df, x="group1", y="group2", size="meandiff", color="reject",
                                   title="Результаты теста Тьюки")
            plots["tukey"] = fig_tukey.to_html(full_html=False, include_plotlyjs='cdn')

            return render_template(
                "tukey.html",
                active_page="tukey",
                anova_table=anova_table.to_html(classes="table table-bordered table-sm"),
                tukey_result=tukey_result.as_html(),
                plots=plots
            )

        return render_template("tukey.html", active_page="tukey")

    except Exception as e:
        return render_template("tukey.html", error=str(e), active_page="tukey")




# === Mann–Whitney ===
@app.route('/mannwhitney', methods=['GET', 'POST'])
def mannwhitney_test():
    u_stat = None
    p_value = None
    conclusion = None
    plots = {}
    error = None

    # Локальный парсер данных
    def _parse_input(s):
        if not s:
            return []
        try:
            return [float(x) for x in s.replace(',', ' ').split() if x.strip()]
        except Exception:
            return []

    if request.method == 'POST':
        try:
            g1 = _parse_input(request.form.get('group1', ''))
            g2 = _parse_input(request.form.get('group2', ''))

            # Проверки
            if not g1 or not g2:
                error = "Введите данные для обеих групп."
                flash(error, "error")
            elif len(g1) < 2 or len(g2) < 2:
                error = "Нужно минимум по 2 значения в каждой группе."
                flash(error, "error")
            else:
                # Сам тест
                res = mannwhitneyu(g1, g2, alternative='two-sided')
                u_stat, p_value = float(res.statistic), float(res.pvalue)

                conclusion = (
                    "Различия статистически значимы (p < 0.05)."
                    if p_value < 0.05
                    else "Статистически значимых различий не обнаружено."
                )

                # DataFrame для графиков
                df = pd.DataFrame({
                    "Value": g1 + g2,
                    "Group": ["Группа 1"] * len(g1) + ["Группа 2"] * len(g2)
                })

                # 4 графика Plotly
                plots['boxplot'] = px.box(
                    df, x="Group", y="Value", points="all", title="Boxplot по группам"
                ).to_html(full_html=False, include_plotlyjs='cdn')

                plots['hist'] = px.histogram(
                    df, x="Value", color="Group", barmode="overlay",
                    nbins=20, opacity=0.6, title="Наложенные гистограммы"
                ).to_html(full_html=False, include_plotlyjs='cdn')

                plots['violin'] = px.violin(
                    df, x="Group", y="Value", box=True, points="all",
                    title="Violin plot (форма распределения)"
                ).to_html(full_html=False, include_plotlyjs='cdn')

                plots['strip'] = px.strip(
                    df, x="Group", y="Value", stripmode="overlay",
                    title="Точки выборки (strip plot)"
                ).to_html(full_html=False, include_plotlyjs='cdn')

        except Exception as e:
            error = str(e)
            flash(error, "error")

    return render_template(
        "mannwhitney.html",
        u_stat=u_stat,
        p_value=p_value,
        conclusion=conclusion,
        plots=plots,
        error=error,
        active_page="mannwhitney"
    )



# === Fisher ===
@app.route("/fisher", methods=["GET", "POST"])
def fisher_test():
    odds_ratio = None
    p_value = None
    error = None
    plots = {}

    if request.method == "POST":
        try:
            table_data = request.form.get("table", "")
            table = [[int(num) for num in row.split()] for row in table_data.strip().split("\n") if row.strip()]
            if len(table) != 2 or any(len(row) != 2 for row in table):
                error = "Введите 2 строки по 2 числа"
            else:
                odds_ratio, p_value = stats.fisher_exact(table)

                # DataFrame
                df = pd.DataFrame(table, index=["Группа 1", "Группа 2"], columns=["Категория A", "Категория B"])

                # 1. Heatmap
                fig_heat = px.imshow(df, text_auto=True, color_continuous_scale="RdBu", title="Тепловая карта 2×2")
                plots["heatmap"] = fig_heat.to_html(full_html=False)

                # 2. Bar chart
                fig_bar = px.bar(df.reset_index().melt(id_vars="index"), x="index", y="value",
                                 color="variable", barmode="group", title="Bar Chart по категориям")
                plots["bar"] = fig_bar.to_html(full_html=False)

                # 3. Гистограмма всех значений
                fig_hist = px.histogram(df.values.flatten(), nbins=5, title="Гистограмма значений таблицы")
                plots["hist"] = fig_hist.to_html(full_html=False)

                # 4. Мозаичный график
                df_melt = df.reset_index().melt(id_vars="index", value_name="count")
                fig_mosaic = px.treemap(df_melt, path=["index", "variable"], values="count",
                                        title="Мозаичный график распределения")
                plots["mosaic"] = fig_mosaic.to_html(full_html=False)

        except Exception as e:
            error = f"Ошибка: {str(e)}"

    return render_template("fisher.html", active_page="fisher",
                           odds_ratio=odds_ratio, p_value=p_value, error=error, plots=plots)






# === Levene ===
@app.route("/levene", methods=["GET", "POST"])
def levene_test():
    stat = None
    p_value = None
    conclusion = None
    plots = {}
    error = None

    if request.method == "POST":
        try:
            group1_str = request.form.get("group1", "")
            group2_str = request.form.get("group2", "")
            group1 = [float(x.strip()) for x in group1_str.split(",") if x.strip()]
            group2 = [float(x.strip()) for x in group2_str.split(",") if x.strip()]

            if len(group1) < 2 or len(group2) < 2:
                raise ValueError("В каждой группе должно быть хотя бы 2 числа")

            stat, p_value = stats.levene(group1, group2)

            if p_value < 0.05:
                conclusion = "Дисперсии статистически различаются (p < 0.05)"
            else:
                conclusion = "Дисперсии не различаются статистически значимо (p >= 0.05)"

            # Подготовка данных для графиков
            df = pd.DataFrame({
                "Value": group1 + group2,
                "Group": ["Группа 1"] * len(group1) + ["Группа 2"] * len(group2)
            })

            # Boxplot
            fig_box = px.box(df, x="Group", y="Value", points="all", title="Boxplot")
            plots["boxplot"] = fig_box.to_html(full_html=False, include_plotlyjs='cdn')

            # Histogram
            fig_hist = px.histogram(df, x="Value", color="Group", nbins=10, barmode="overlay", title="Гистограмма")
            plots["histogram"] = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')

            # KDE (плотность распределения)
            fig_kde = go.Figure()
            for grp, data in df.groupby("Group")["Value"]:
                kde_x = np.linspace(min(data), max(data), 200)
                kde_y = stats.gaussian_kde(data)(kde_x)
                fig_kde.add_trace(go.Scatter(x=kde_x, y=kde_y, mode='lines', name=grp))
            fig_kde.update_layout(title="Плотность распределения (KDE)")
            plots["kde"] = fig_kde.to_html(full_html=False, include_plotlyjs='cdn')

            # QQ-Plot
            fig_qq = go.Figure()
            for grp, data in df.groupby("Group")["Value"]:
                sm.qqplot(np.array(data), line='s')
            import io
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            import base64
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            fig_qq_html = f'<img src="data:image/png;base64,{img_base64}" />'
            plots["qqplot"] = fig_qq_html
            plt.close()

        except Exception as e:
            error = str(e)

    return render_template(
        "levene.html",
        active_page="levene",
        stat=stat,
        p_value=p_value,
        conclusion=conclusion,
        plots=plots,
        error=error
    )



# === Linear Regression ===
@app.route("/linear_regression", methods=["GET", "POST"])
def linear_regression_test():
    plots = {}
    slope = intercept = r_value = p_value = std_err = None
    error = None

    if request.method == "POST":
        try:
            # Получаем данные из формы
            x_data_str = request.form.get("x_data", "")
            y_data_str = request.form.get("y_data", "")
            x_data = [float(i) for i in x_data_str.replace(",", " ").split()]
            y_data = [float(i) for i in y_data_str.replace(",", " ").split()]

            if len(x_data) != len(y_data) or len(x_data) < 2:
                raise ValueError("Введите одинаковое количество X и Y (минимум 2 значения).")

            # Линейная регрессия
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            y_pred = [slope * xi + intercept for xi in x_data]
            residuals = [yi - ypi for yi, ypi in zip(y_data, y_pred)]

            df = pd.DataFrame({
                "X": x_data,
                "Y": y_data,
                "Predicted": y_pred,
                "Residuals": residuals
            })

            # 1. Scatter + линия регрессии
            plots['scatter'] = px.scatter(df, x="X", y="Y", title="Линейная регрессия: точки и линия")\
                .add_scatter(x=df["X"], y=df["Predicted"], mode="lines", name="Regression Line")\
                .to_html(full_html=False, include_plotlyjs='cdn')

            # 2. Остатки vs Предсказанные значения
            plots['residuals'] = px.scatter(df, x="Predicted", y="Residuals",
                                            title="Остатки vs Предсказанные значения")\
                .to_html(full_html=False, include_plotlyjs='cdn')

            # 3. QQ-plot остатков
            qq_fig = go.Figure()
            (osm, osr), _ = stats.probplot(residuals)
            qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Residuals'))
            qq_fig.add_trace(go.Scatter(x=osm, y=osm, mode='lines', name='Ideal Line'))
            qq_fig.update_layout(title="QQ-Plot остатков")
            plots['qqplot'] = qq_fig.to_html(full_html=False, include_plotlyjs='cdn')

            # 4. Гистограмма остатков
            plots['hist_residuals'] = px.histogram(df, x="Residuals", nbins=20, title="Гистограмма остатков")\
                .to_html(full_html=False, include_plotlyjs='cdn')

        except Exception as e:
            error = str(e)

    return render_template("linear_regression.html",
                           slope=slope, intercept=intercept, r_value=r_value,
                           p_value=p_value, std_err=std_err, plots=plots, error=error,
                           active_page="linear_regression")


# === Polynomial Regression ===
@app.route('/polynomial_regression', methods=['GET', 'POST'])
def polynomial_regression_test():
    coefficients = None
    intercept = None
    r2_score_val = None
    plots = {}
    error = None

    if request.method == 'POST':
        try:
            x_data = parse_input_data(request.form.get('x_data', ''))
            y_data = parse_input_data(request.form.get('y_data', ''))
            degree = int(request.form.get('degree', 2))

            if len(x_data) != len(y_data) or len(x_data) < 2:
                error = "Количество X и Y должно совпадать и быть >= 2."
            else:
                import numpy as np
                import pandas as pd
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                import plotly.express as px
                import plotly.graph_objects as go
                import statsmodels.api as sm
                import io, base64

                # --- фикс для macOS и многопоточности ---
                import matplotlib
                matplotlib.use("Agg")  # без GUI backend
                import matplotlib.pyplot as plt
                # ----------------------------------------

                X = np.array(x_data).reshape(-1, 1)
                y = np.array(y_data)

                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X)

                model = LinearRegression()
                model.fit(X_poly, y)

                y_pred = model.predict(X_poly)

                coefficients = [float(c) for c in model.coef_]
                intercept = float(model.intercept_)
                r2_score_val = float(r2_score(y, y_pred))

                df = pd.DataFrame({"X": x_data, "Y": y_data, "Y_pred": y_pred})

                # 1. Scatter + Polynomial line (отсортированная линия)
                sort_idx = np.argsort(df["X"])
                fig_fit = go.Figure()
                fig_fit.add_trace(go.Scatter(x=df["X"], y=df["Y"], mode='markers', name='Данные'))
                fig_fit.add_trace(go.Scatter(
                    x=df["X"].iloc[sort_idx],
                    y=df["Y_pred"].iloc[sort_idx],
                    mode='lines',
                    name='Модель'
                ))
                plots['fit'] = fig_fit.to_html(full_html=False, include_plotlyjs='cdn')

                # 2. Residuals plot
                residuals = y - y_pred
                fig_resid = px.scatter(x=df["X"], y=residuals, title="График остатков")
                fig_resid.update_layout(xaxis_title="X", yaxis_title="Остатки")
                plots['residuals'] = fig_resid.to_html(full_html=False, include_plotlyjs='cdn')

                # 3. Residuals histogram
                fig_hist = px.histogram(residuals, nbins=20, title="Гистограмма остатков")
                plots['residuals_hist'] = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')

                # 4. QQ-plot (сохранение через объект Figure)
                qq_fig = sm.qqplot(residuals, line='s')
                buf = io.BytesIO()
                qq_fig.savefig(buf, format="png")
                buf.seek(0)
                plots['qqplot'] = f"<img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}'/>"
                plt.close(qq_fig)

        except Exception as e:
            error = str(e)

    return render_template(
        "polynomial_regression.html",
        coefficients=coefficients,
        intercept=intercept,
        r2_score_val=r2_score_val,
        plots=plots,
        error=error,
        active_page="polynomial_regression"
    )



# === Logistic Regression ===
@app.route("/logistic_regression", methods=["GET", "POST"])
def logistic_regression_test():
    coef = None
    intercept = None
    accuracy = None
    plots = {}
    error = None

    # показываем форму при GET
    if request.method == "GET":
        return render_template("logistic_regression.html",
                               coef=coef, intercept=intercept, accuracy=accuracy,
                               plots=plots, error=error, active_page="logistic_regression")

    # POST — обрабатываем ввод
    try:
        # используем вашу парсер-функцию (parse_input_data)
        X_raw = request.form.get("x_values", "")
        y_raw = request.form.get("y_values", "")

        X_list = parse_input_data(X_raw)
        y_list = parse_input_data(y_raw)

        if not X_list or not y_list:
            error = "Введите данные для X и y."
        elif len(X_list) != len(y_list):
            error = "Длины X и y должны совпадать."
        else:
            X = np.array(X_list).reshape(-1, 1)
            y = np.array([int(v) for v in y_list])  # ожидаем 0/1 или 0/1-like

            # Проверка меток (0/1)
            unique_y = np.unique(y)
            if not set(unique_y).issubset({0, 1}):
                error = "y должен содержать метки 0 или 1 (можно разделять пробелом или запятой)."
            else:
                model = LogisticRegression(max_iter=1000)
                model.fit(X, y)

                coef = float(model.coef_[0][0])
                intercept = float(model.intercept_[0])

                y_pred = model.predict(X)
                accuracy = float(accuracy_score(y, y_pred))

                # ROC
                y_prob = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                roc_fig.update_layout(title="ROC-кривая", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                plots['roc'] = roc_fig.to_html(full_html=False, include_plotlyjs='cdn')

                # Гистограмма вероятностей по классам
                df_probs = pd.DataFrame({"prob": y_prob, "y": y.astype(str)})
                hist_fig = px.histogram(df_probs, x="prob", color="y", nbins=20, barmode="overlay",
                                        title="Гистограмма предсказанных вероятностей (класс 1 vs 0)")
                plots['hist'] = hist_fig.to_html(full_html=False, include_plotlyjs='cdn')

                # Scatter + модельная линия (как у линейной, но для вероятности)
                scatter_fig = go.Figure()
                scatter_fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode="markers", name="Истинные метки"))
                # Для линии предсказанных вероятностей упорядочим по X
                order_idx = np.argsort(X.flatten())
                x_line = X.flatten()[order_idx]
                prob_line = y_prob[order_idx]
                scatter_fig.add_trace(go.Scatter(x=x_line, y=prob_line, mode="lines", name="Вероятность класса 1"))
                scatter_fig.update_layout(title="Данные и предсказанные вероятности", xaxis_title="X", yaxis_title="y / prob")
                plots['scatter'] = scatter_fig.to_html(full_html=False, include_plotlyjs='cdn')

                # Плот распределения вероятностей (kde-стиль через histogram with density)
                kde_fig = px.histogram(df_probs, x="prob", color="y", nbins=40, marginal="violin",
                                       title="Распределение вероятностей с violin (marginal)")
                plots['kde'] = kde_fig.to_html(full_html=False, include_plotlyjs='cdn')

    except Exception as e:
        error = str(e)

    return render_template("logistic_regression.html",
                           coef=coef, intercept=intercept, accuracy=accuracy,
                           plots=plots, error=error, active_page="logistic_regression")





if __name__ == "__main__":
    app.run(
        host="0.0.0.0",  # доступ с любого устройства в сети
        port=5000,
        debug=True,      # автоматический перезапуск при ошибках и изменениях
        use_reloader=True
    )



