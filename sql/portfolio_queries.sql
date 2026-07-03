-- ============================================
-- Credit Risk Portfolio SQL Queries
-- ============================================

-- 1. Portfolio default rate by risk segment
SELECT
    risk_segment,
    COUNT(*) AS customer_count,
    SUM(is_default) AS defaults,
    AVG(is_default) AS default_rate,
    AVG(default_probability) AS avg_pd,
    SUM(expected_loss) AS total_expected_loss
FROM feature_table
GROUP BY risk_segment
ORDER BY default_rate DESC;


-- 2. Credit utilization distribution
SELECT
    CASE
        WHEN avg_utilization < 0.3 THEN 'Low (<30%)'
        WHEN avg_utilization < 0.6 THEN 'Medium (30-60%)'
        WHEN avg_utilization < 0.8 THEN 'High (60-80%)'
        ELSE 'Very High (>80%)'
    END AS utilization_bucket,
    COUNT(*) AS count,
    AVG(default_probability) AS avg_pd,
    AVG(expected_loss) AS avg_el
FROM feature_table
GROUP BY utilization_bucket
ORDER BY avg_pd;


-- 3. Top delinquent customers with highest exposure
SELECT
    customer_id,
    credit_limit,
    current_balance,
    delinquency_severity,
    late_payment_count,
    default_probability,
    expected_loss
FROM feature_table
WHERE delinquency_severity >= 2
ORDER BY expected_loss DESC
LIMIT 50;


-- 4. Monthly default trend
SELECT
    DATE_TRUNC('month', statement_date) AS month,
    COUNT(DISTINCT customer_id) AS active_customers,
    SUM(is_default) AS new_defaults,
    AVG(default_probability) AS avg_pd,
    SUM(statement_balance) AS total_balance
FROM statements
    JOIN defaults USING (customer_id)
GROUP BY month
ORDER BY month;


-- 5. Payment behavior vs default
SELECT
    CASE
        WHEN avg_payment_to_balance >= 0.9 THEN 'Full payer'
        WHEN avg_payment_to_balance >= 0.5 THEN 'Partial payer'
        WHEN avg_payment_to_balance >= 0.1 THEN 'Minimum payer'
        ELSE 'Non-payer'
    END AS payer_type,
    COUNT(*) AS count,
    AVG(is_default) AS default_rate,
    AVG(late_payment_count) AS avg_late_payments
FROM feature_table
GROUP BY payer_type
ORDER BY default_rate;


-- 6. Expected loss by state
SELECT
    state,
    COUNT(*) AS customers,
    SUM(expected_loss) AS total_el,
    AVG(default_probability) AS avg_pd,
    AVG(credit_score) AS avg_credit_score
FROM feature_table
    JOIN customers USING (customer_id)
GROUP BY state
ORDER BY total_el DESC;


-- 7. Feature correlation with defaults
SELECT
    CORR(avg_utilization, CAST(is_default AS FLOAT)) AS util_corr,
    CORR(late_payment_count, CAST(is_default AS FLOAT)) AS late_pay_corr,
    CORR(income_to_debt_ratio, CAST(is_default AS FLOAT)) AS income_debt_corr,
    CORR(delinquency_severity, CAST(is_default AS FLOAT)) AS delinq_corr,
    CORR(credit_score, CAST(is_default AS FLOAT)) AS credit_score_corr
FROM feature_table
    JOIN customers USING (customer_id);
