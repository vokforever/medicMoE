-- SQL скрипт для создания таблицы структурированных данных анализов
-- Выполнять в Supabase SQL Editor
-- Этот скрипт создаст таблицу для хранения структурированных данных анализов пациентов

-- Проверяем и создаем расширения, если их нет
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Удаляем существующую таблицу (если есть) для пересоздания
DROP TABLE IF EXISTS doc_structured_test_results CASCADE;

-- Создание таблицы для структурированных данных анализов
CREATE TABLE doc_structured_test_results (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    test_name TEXT NOT NULL,
    result TEXT NOT NULL,
    reference_values TEXT,
    units TEXT,
    test_date DATE,
    test_system TEXT,
    equipment TEXT,
    notes TEXT,
    source_record_id BIGINT REFERENCES doc_medical_records(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Добавляем индексы для быстрого поиска
CREATE INDEX idx_doc_structured_test_results_user_id ON doc_structured_test_results(user_id);
CREATE INDEX idx_doc_structured_test_results_test_name ON doc_structured_test_results(test_name);
CREATE INDEX idx_doc_structured_test_results_test_date ON doc_structured_test_results(test_date);

-- Добавляем уникальное ограничение для предотвращения дубликатов (если не существует)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'unique_doc_structured_test_results_user_test_date' 
        AND conrelid = 'doc_structured_test_results'::regclass
    ) THEN
        ALTER TABLE doc_structured_test_results ADD CONSTRAINT unique_doc_structured_test_results_user_test_date 
        UNIQUE (user_id, test_name, test_date);
    END IF;
END $$;

-- Создаем функцию для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Создаем триггер для автоматического обновления updated_at
CREATE TRIGGER update_doc_structured_test_results_updated_at 
    BEFORE UPDATE ON doc_structured_test_results 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Комментарии к таблице и столбцам
COMMENT ON TABLE doc_structured_test_results IS 'Структурированные данные анализов пациентов';
COMMENT ON COLUMN doc_structured_test_results.user_id IS 'UUID пользователя';
COMMENT ON COLUMN doc_structured_test_results.test_name IS 'Название анализа (например, "Anti-HCV total (анти-HCV)")';
COMMENT ON COLUMN doc_structured_test_results.result IS 'Результат анализа';
COMMENT ON COLUMN doc_structured_test_results.reference_values IS 'Референсные значения (норма)';
COMMENT ON COLUMN doc_structured_test_results.units IS 'Единицы измерения';
COMMENT ON COLUMN doc_structured_test_results.test_date IS 'Дата сдачи анализа';
COMMENT ON COLUMN doc_structured_test_results.test_system IS 'Тест-система';
COMMENT ON COLUMN doc_structured_test_results.equipment IS 'Оборудование';
COMMENT ON COLUMN doc_structured_test_results.notes IS 'Примечания';
COMMENT ON COLUMN doc_structured_test_results.source_record_id IS 'ID исходной медицинской записи';

-- Сообщение об успешном выполнении
SELECT 'Table doc_structured_test_results created successfully!' as status;

-- ПРИМЕЧАНИЕ: После выполнения этого скрипта система сможет:
-- 1. Сохранять структурированные данные анализов
-- 2. Автоматически обновлять время изменения записей
-- 3. Предотвращать дубликаты анализов одного типа в одну дату
-- 4. Быстро искать анализы по пользователю, названию и дате
