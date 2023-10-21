from functions import *


if 'show_corr_map' not in st.session_state:
	clean_vars()

st.set_page_config(
    page_title='DSS for Uplift Modeling with DDR method',
    page_icon='🎓',
    layout='wide'
)

# st.write(st.session_state)
st.write("# Decision Support System for Uplift Modeling with Dependent Data Representation method")
st.write('## 1. Огляд завантажених даних')

with st.sidebar:
    st.write('# 1. Завантаження даних')
    st.write('## 1.1. Виберіть спосіб розбиття даних на тренувальні та тестові')
    train_test_options = [
        'Завантажити один файл з даними та розбити їх на тренувальні та тестові в програмі',
        'Завантажити два окремих файли з тренувальними та тестувальними даними відповідно'
    ]
    split_method = st.radio('Cпосіб розбиття даних на тренувальні та тестові', options=train_test_options, index=None)

if split_method == 'Завантажити один файл з даними та розбити їх на тренувальні та тестові в програмі':
    with st.sidebar:
        st.write('## 1.2. Виберіть файл з даними')
        data_file = st.file_uploader("Виберіть файл з даними", type=["xlsx", "csv", "parquet"])
        train_file, test_file = None, None

    if data_file is not None:
        data = file_to_df(data_file)
        st.write('Розмір оригінальних даних:', data.shape)
        with st.sidebar:
            st.write('## 1.3. Виберіть значення параметрів для розбиття даних')
            split_rand_seed = st.number_input('Сід генератору випадкових чисел для розбиття даних', min_value=1, value=42)
            split_test_size = st.number_input('Відсоток тренувальних даних', min_value=0, max_value=100, value=20) / 100
        data_train, data_test = train_test_split(data, test_size=split_test_size, random_state=split_rand_seed)

if split_method == 'Завантажити два окремих файли з тренувальними та тестувальними даними відповідно':
    with st.sidebar:
        st.header('1.2. Виберіть файли з даними')
        data_file = None
        train_file = st.file_uploader("Виберіть файл з тренувальними даними", type=["xlsx", "csv", "parquet"])
        test_file = st.file_uploader("Виберіть файл з тестовими даними", type=["xlsx", "csv", "parquet"])
 
    if train_file is not None and test_file is not None:
        data_train = file_to_df(train_file) 
        data_test = file_to_df(test_file) 

if split_method is not None:
    if data_file is not None or (train_file is not None and test_file is not None):
        c1, c2 = st.columns(2) 
        with c1:
            st.write('### Тренувальні дані')
            data_info(data_train)
        with c2:
            st.write('### Тестові дані')
            data_info(data_test)
        st.divider()

        
        st.write('## 2. Коефіціенти рангової кореляції Спірмена в тренувальних даних')
        with st.sidebar:
            st.divider()
            st.write('# 2. Аналіз тренувальних даних')
            st.write('## 2.1. Виберіть які ознаки відображають:')
            id_col = st.selectbox('Ідентифікатор користувача', list(data_train.columns), index=None)
            
            if id_col is not None:
                treatment_col = st.selectbox('Ознака комунікації', list(data_train.drop(columns=[id_col]).columns), index=None)
                if treatment_col is not None:
                    response_col = st.selectbox('Ознака виконання цільової дії', list(data_train.drop(columns=[id_col, treatment_col]).columns), index=None)
            
        if id_col is not None and treatment_col is not None and response_col is not None:
            with st.sidebar:
                st.write('## 2.2. Видалення ознак та генерування кореляційної матриці')
                del_required = st.checkbox('Видалення ознак необхідне')
                if del_required:
                    ban_cols = st.multiselect('Ознаки на видалення', list(data_train.drop(columns=[id_col, treatment_col, response_col]).columns))
                    # ban_cols = st.multiselect('Ознаки на видалення', list(data_train.drop(columns=[id_col, treatment_col, response_col]).columns),
                    #         default = ['X_21', 'X_25', 'X_45', 'X_43', 'X_29', 'X_1', 'X_11', 'X_42', 'X_22', 'X_2', 'X_8', 'X_15', 'X_10', 'X_3', 'X_17']
                    #     )
                else:
                    ban_cols = []
                
                generate_corr_map_clicked = st.button('Внести зміни та згенерувати матрицю')

            
            cols = list(data_train.drop(columns=[id_col, treatment_col]+ban_cols).columns)
            if generate_corr_map_clicked:
                with st.spinner('Розрахунок коефіцієнтів кореляції в процесі...'):
                    st.session_state.show_corr_map = True
                    corr_comp = data_train[cols].corr(method='spearman')[response_col].reset_index().merge(
                            data_train.loc[data_train[treatment_col]==0, cols].astype(float).corr(method='spearman')[response_col].reset_index(), on='index', suffixes=('', '_treatment')
                        ).merge(
                            data_train.loc[data_train[treatment_col]==1, cols].astype(float).corr(method='spearman')[response_col].reset_index(), on='index', suffixes=('', '_control')
                        ).sort_values(by=response_col, ascending=False, key = lambda x: abs(x)).round(4).set_index('index').iloc[1:].T
                    st.session_state.corr_response = corr_comp
                    st.session_state.corr_map = corr_heatmap(data_train[cols])
            
            if st.session_state.show_corr_map:
                st.write('### Кореляційна матриця')
                st.plotly_chart(st.session_state.corr_map, theme=None)
                st.write('### Коефіцієнти кореляції з цільовою змінною')
                st.write('*(відсортовані за спаданням модулю значення)*')
                st.write(st.session_state.corr_response)
                st.divider()


                st.write('## 3. Побудова та оцінка якості моделі')
                X_train, y_train, t_train = get_x_y_t(data_train.drop(columns=ban_cols), id_col, response_col, treatment_col)
                X_test, y_test, t_test = get_x_y_t(data_test.drop(columns=ban_cols), id_col, response_col, treatment_col)
                with st.sidebar:
                    st.divider()
                    st.write('# 3. Побудова моделі')
                    st.write('## 3.1. Гіперпараметри моделі')
                    model_rand_seed = st.number_input('Сід генератору випадкових чисел для побудови моделі', min_value=1, value=42)
                    model_ddr_feature = st.selectbox('Група, результати моделі якої будуть ознакою для моделі іншої групи', ['treatment', 'control'])
                    train_model_clicked = st.button('Почати тренування')

                
                if train_model_clicked:
                    with st.spinner('Тренування моделі в процесі...'):
                        st.session_state.last_model = TwoModelsDDR(
                            classifier_trmt=XGBClassifier(random_state=model_rand_seed), 
                            classifier_ctrl=XGBClassifier(random_state=model_rand_seed),
                            ddr_feature=model_ddr_feature
                        )
                        st.session_state.last_model.fit(X_train, y_train, t_train)
                        st.session_state.last_model_uplift = st.session_state.last_model.predict(X_test)

                if st.session_state.last_model is not None:
                    st.success('Модель успішно натреновано. У боковому вікні зліва можна налаштувати відображення графіків якості моделі.', icon="✅")
                    with st.sidebar:
                        st.write('## 3.2. Параметри оцінки якості моделі')
                        st.write('### 3.2.1. Uplift by percentile')
                        split_test_size = st.number_input('N для метрики "Uplift at top N%"', min_value=0, max_value=100, value=30) / 100
                        ubp_type = st.radio('Спосіб відображення "Uplift by percentile"', options=['barchart', 'table'])
                        st.write('### 3.2.2. Qini крива')
                        qini_x_type = st.radio('Вісь Х в графіку Qini кривої', options=['Percent of users', 'Number of users']) == 'Percent of users'
                        perfect_qini_required = st.checkbox('Відображати Qini криву ідеальної моделі')
                    c31, c32 = st.columns(2) 
                    with c31:
                        st.pyplot(plot_uplift_by_percentile_barchart(t_test, y_test, st.session_state.last_model_uplift, top_perc=split_test_size))
                    with c32:
                        st.pyplot(plot_qini_curve(t_test, y_test, st.session_state.last_model_uplift, normalize_n=qini_x_type, plot_perfect=perfect_qini_required))
                    st.divider()
                
                    st.write('## 4. Оцінка впливу акції на нових користувачів')
                    with st.sidebar:
                        st.write('# 4. Оцінка впливу акції на нових користувачів')
                        st.write('## 4.1. Виберіть файл з даними')
                        new_users_file = st.file_uploader("Виберіть файл з даними про нових користувачів", type=["xlsx", "csv", "parquet"])

                    if new_users_file is not None:
                        new_users = file_to_df(new_users_file) 
                        st.write('### Огляд завантажених даних')
                        st.write(new_users)
                        with st.spinner('Розрахунок оцінок впливу акції в процесі...'):
                            X_new_users = new_users.drop(columns=ban_cols+[id_col])
                            uplift_new_users = st.session_state.last_model.predict(X_new_users)
                            uplift_new_users = pd.concat([new_users[id_col], pd.Series(uplift_new_users, name='predicted_uplift')], axis=1)
                        st.write('### Результат моделі')
                        info_msg('Результат моделі можна завантажити в боковому вікні зліва.')
                        st.write(uplift_new_users)
                        with st.sidebar:
                            st.write('## 4.2. Завантаження результатів')
                            st.download_button(
                                label="📥 Завантажити в .csv форматі",
                                data=uplift_new_users.to_csv(),
                                file_name='large_df.csv',
                                mime='text/csv',
                            )


                    else:
                        info_msg('Виберіть файл з даними у боковому вікні зліва.')


                else:
                    info_msg('Після внесення необхідних змін в дані, оберіть гіперпараметри моделі та натисність кнопку "Почати тренування" у боковому вікні зліва.')
                    clean_model_vars()

            
            else:
                info_msg('Виберіть зайві ознаки на видалення (якщо такі присутні) у боковому вікні зліва та натисніть кнопку "Внести зміни та згенерувати матрицю", щоб видалити обрані ознаки та порахувати кореляційні коефіцієнти.')
                clean_vars()

        else:
            info_msg('Виберіть ознаки, які відображають ідентифікатор користувача, ознаку комунікації та ознаку виконання цільової дії, у боковому вікні зліва щоб продовжити.')
            clean_vars()

    else:
        info_msg('Виберіть всі файли з даними у боковому вікні зліва щоб продовжити.')
        clean_vars()


if split_method is None:
    info_msg('Виберіть спосіб розбиття даних у боковому вікні зліва щоб продовжити.')
    clean_vars()
