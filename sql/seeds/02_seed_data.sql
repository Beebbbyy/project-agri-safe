-- Project Agri-Safe Seed Data
-- Sample data for development and testing

-- =====================================================
-- Seed Data: Philippine Regions
-- =====================================================
INSERT INTO regions (region_name, region_code, province, latitude, longitude) VALUES
    -- Luzon
    ('Ilocos Region', 'Region I', 'Ilocos Norte', 18.1635, 120.7146),
    ('Ilocos Region', 'Region I', 'Pangasinan', 15.8949, 120.2863),
    ('Cagayan Valley', 'Region II', 'Cagayan', 17.6132, 121.7270),
    ('Cagayan Valley', 'Region II', 'Isabela', 16.9754, 121.8107),
    ('Central Luzon', 'Region III', 'Bulacan', 14.7942, 120.8906),
    ('Central Luzon', 'Region III', 'Nueva Ecija', 15.5784, 121.1113),
    ('Central Luzon', 'Region III', 'Pampanga', 15.0794, 120.6200),
    ('CALABARZON', 'Region IV-A', 'Cavite', 14.4791, 120.8970),
    ('CALABARZON', 'Region IV-A', 'Laguna', 14.2691, 121.4113),
    ('CALABARZON', 'Region IV-A', 'Batangas', 13.7565, 121.0583),
    ('MIMAROPA', 'Region IV-B', 'Occidental Mindoro', 13.1000, 120.7650),
    ('MIMAROPA', 'Region IV-B', 'Oriental Mindoro', 13.0000, 121.5000),
    ('Bicol Region', 'Region V', 'Camarines Sur', 13.5224, 123.1784),
    ('Bicol Region', 'Region V', 'Albay', 13.1391, 123.7256),

    -- Visayas
    ('Western Visayas', 'Region VI', 'Iloilo', 10.7202, 122.5621),
    ('Western Visayas', 'Region VI', 'Negros Occidental', 10.6590, 122.9738),
    ('Central Visayas', 'Region VII', 'Cebu', 10.3157, 123.8854),
    ('Central Visayas', 'Region VII', 'Bohol', 9.8500, 124.1435),
    ('Eastern Visayas', 'Region VIII', 'Leyte', 11.2500, 124.8333),
    ('Eastern Visayas', 'Region VIII', 'Samar', 11.5804, 125.0319),

    -- Mindanao
    ('Zamboanga Peninsula', 'Region IX', 'Zamboanga del Norte', 8.5500, 123.3500),
    ('Zamboanga Peninsula', 'Region IX', 'Zamboanga del Sur', 7.8381, 123.2956),
    ('Northern Mindanao', 'Region X', 'Bukidnon', 8.0542, 124.9252),
    ('Northern Mindanao', 'Region X', 'Misamis Oriental', 8.5050, 124.6169),
    ('Davao Region', 'Region XI', 'Davao del Sur', 6.7763, 125.3453),
    ('Davao Region', 'Region XI', 'Davao del Norte', 7.5619, 125.6533),
    ('SOCCSKSARGEN', 'Region XII', 'South Cotabato', 6.3162, 124.7936),
    ('SOCCSKSARGEN', 'Region XII', 'Sultan Kudarat', 6.5044, 124.4169),
    ('Caraga', 'Region XIII', 'Agusan del Norte', 8.9472, 125.5325),
    ('Caraga', 'Region XIII', 'Agusan del Sur', 8.5617, 125.9750)
ON CONFLICT (region_name) DO NOTHING;

-- =====================================================
-- Seed Data: Crop Types
-- =====================================================
INSERT INTO crop_types (crop_name, crop_category, typical_growth_days, min_growth_days, max_growth_days,
                        optimal_temp_min, optimal_temp_max, water_requirement, flood_tolerance, description) VALUES
    -- Rice varieties
    ('Rice - Irrigated', 'rice', 120, 110, 130, 20.0, 35.0, 'high', 'medium',
     'Traditional irrigated rice variety commonly grown in Philippines'),
    ('Rice - Rainfed Lowland', 'rice', 130, 120, 140, 20.0, 35.0, 'high', 'medium',
     'Rice variety dependent on rainfall, grown in lowland areas'),
    ('Rice - Upland', 'rice', 110, 100, 120, 20.0, 30.0, 'medium', 'low',
     'Rice variety grown in upland areas without standing water'),

    -- Corn varieties
    ('Corn - White', 'corn', 100, 90, 110, 21.0, 30.0, 'medium', 'low',
     'White corn variety, staple food in many Filipino regions'),
    ('Corn - Yellow', 'corn', 95, 85, 105, 21.0, 30.0, 'medium', 'low',
     'Yellow corn variety, used for animal feed and human consumption'),

    -- Vegetables
    ('Tomato', 'vegetables', 70, 60, 90, 18.0, 27.0, 'medium', 'low',
     'Common vegetable crop, sensitive to excessive rainfall'),
    ('Eggplant', 'vegetables', 75, 65, 90, 21.0, 30.0, 'medium', 'low',
     'Popular vegetable in Filipino cuisine, prefers warm climate'),
    ('Bitter Gourd (Ampalaya)', 'vegetables', 55, 50, 65, 24.0, 27.0, 'medium', 'low',
     'Traditional Filipino vegetable, heat-tolerant'),
    ('String Beans (Sitaw)', 'vegetables', 50, 45, 60, 18.0, 30.0, 'medium', 'low',
     'Fast-growing vegetable, popular in Filipino dishes'),
    ('Cabbage', 'vegetables', 70, 60, 90, 15.0, 25.0, 'medium', 'low',
     'Cool-season crop, grown in highland areas'),
    ('Lettuce', 'vegetables', 45, 40, 60, 15.0, 20.0, 'medium', 'low',
     'Cool-season leafy vegetable'),
    ('Onion', 'vegetables', 100, 90, 120, 13.0, 24.0, 'medium', 'low',
     'Bulb vegetable, requires well-drained soil'),
    ('Garlic', 'vegetables', 120, 100, 150, 13.0, 24.0, 'low', 'low',
     'Bulb vegetable, requires dry conditions for harvest'),

    -- Root crops
    ('Sweet Potato (Kamote)', 'root_crops', 120, 100, 150, 21.0, 30.0, 'medium', 'low',
     'Drought-tolerant root crop, staple food in upland areas'),
    ('Cassava', 'root_crops', 300, 240, 365, 25.0, 29.0, 'low', 'medium',
     'Hardy root crop, tolerant to drought and poor soil'),

    -- Fruits
    ('Banana', 'fruits', 270, 240, 365, 26.0, 30.0, 'high', 'medium',
     'Tropical fruit, year-round production'),
    ('Papaya', 'fruits', 270, 240, 300, 21.0, 33.0, 'medium', 'low',
     'Fast-growing tropical fruit tree'),
    ('Mango', 'fruits', 365, 300, 400, 24.0, 30.0, 'low', 'low',
     'Tropical fruit tree, requires dry period for flowering'),

    -- Cash crops
    ('Sugarcane', 'cash_crops', 365, 330, 450, 20.0, 30.0, 'high', 'medium',
     'Major cash crop in Philippines, requires abundant water'),
    ('Coconut', 'cash_crops', 1825, 1460, 2555, 27.0, 30.0, 'medium', 'medium',
     'Major plantation crop, 5-7 years from planting to first harvest'),
    ('Coffee', 'cash_crops', 1095, 730, 1460, 18.0, 28.0, 'medium', 'low',
     'Highland cash crop, 3-4 years from planting to first harvest'),
    ('Cacao', 'cash_crops', 1095, 1000, 1460, 21.0, 32.0, 'high', 'low',
     'Tropical cash crop, 3-4 years from planting to first harvest')
ON CONFLICT (crop_name) DO NOTHING;

-- =====================================================
-- Seed Data: Sample Users (for testing)
-- Password: password123 (hashed with bcrypt)
-- =====================================================
INSERT INTO users (id, username, email, password_hash, full_name, phone_number, is_active, is_verified) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'juan_farmer', 'juan@example.com',
     '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpXKxFgKK',
     'Juan dela Cruz', '+639171234567', true, true),
    ('550e8400-e29b-41d4-a716-446655440002', 'maria_farmer', 'maria@example.com',
     '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpXKxFgKK',
     'Maria Santos', '+639181234567', true, true),
    ('550e8400-e29b-41d4-a716-446655440003', 'pedro_farmer', 'pedro@example.com',
     '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpXKxFgKK',
     'Pedro Reyes', '+639191234567', true, true)
ON CONFLICT (username) DO NOTHING;

-- =====================================================
-- Seed Data: Sample Farms
-- =====================================================
INSERT INTO farms (id, user_id, region_id, farm_name, area_hectares, latitude, longitude, soil_type, irrigation_type) VALUES
    ('660e8400-e29b-41d4-a716-446655440001',
     '550e8400-e29b-41d4-a716-446655440001',
     6, 'Nueva Ecija Rice Farm', 3.5, 15.5784, 121.1113, 'Clay loam', 'Irrigated'),
    ('660e8400-e29b-41d4-a716-446655440002',
     '550e8400-e29b-41d4-a716-446655440001',
     6, 'Nueva Ecija Corn Field', 2.0, 15.5800, 121.1200, 'Sandy loam', 'Rainfed'),
    ('660e8400-e29b-41d4-a716-446655440003',
     '550e8400-e29b-41d4-a716-446655440002',
     7, 'Pampanga Vegetable Farm', 1.5, 15.0794, 120.6200, 'Loam', 'Irrigated'),
    ('660e8400-e29b-41d4-a716-446655440004',
     '550e8400-e29b-41d4-a716-446655440003',
     13, 'Camarines Sur Rice Paddies', 5.0, 13.5224, 123.1784, 'Clay', 'Rainfed')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- Seed Data: Sample Plantings (current season)
-- =====================================================
INSERT INTO plantings (id, farm_id, crop_type_id, planting_date, expected_harvest_date, area_planted_hectares, status) VALUES
    ('770e8400-e29b-41d4-a716-446655440001',
     '660e8400-e29b-41d4-a716-446655440001',
     1, -- Rice - Irrigated
     CURRENT_DATE - INTERVAL '60 days',
     CURRENT_DATE + INTERVAL '60 days',
     3.5, 'active'),
    ('770e8400-e29b-41d4-a716-446655440002',
     '660e8400-e29b-41d4-a716-446655440002',
     4, -- Corn - White
     CURRENT_DATE - INTERVAL '50 days',
     CURRENT_DATE + INTERVAL '50 days',
     2.0, 'active'),
    ('770e8400-e29b-41d4-a716-446655440003',
     '660e8400-e29b-41d4-a716-446655440003',
     6, -- Tomato
     CURRENT_DATE - INTERVAL '40 days',
     CURRENT_DATE + INTERVAL '30 days',
     0.5, 'active'),
    ('770e8400-e29b-41d4-a716-446655440004',
     '660e8400-e29b-41d4-a716-446655440003',
     7, -- Eggplant
     CURRENT_DATE - INTERVAL '45 days',
     CURRENT_DATE + INTERVAL '30 days',
     0.5, 'active'),
    ('770e8400-e29b-41d4-a716-446655440005',
     '660e8400-e29b-41d4-a716-446655440004',
     2, -- Rice - Rainfed
     CURRENT_DATE - INTERVAL '70 days',
     CURRENT_DATE + INTERVAL '60 days',
     5.0, 'active')
ON CONFLICT (id) DO NOTHING;

-- =====================================================
-- Information message
-- =====================================================
DO $$
BEGIN
    RAISE NOTICE 'Seed data loaded successfully!';
    RAISE NOTICE 'Test users created with password: password123';
    RAISE NOTICE '  - juan_farmer / juan@example.com';
    RAISE NOTICE '  - maria_farmer / maria@example.com';
    RAISE NOTICE '  - pedro_farmer / pedro@example.com';
END $$;
