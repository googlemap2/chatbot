
DROP TABLE IF EXISTS "public"."categories";
CREATE TABLE "public"."categories" (
  "id" int4 NOT NULL DEFAULT nextval('categories_id_seq'::regclass),
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "slug" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "image" varchar(500) COLLATE "pg_catalog"."default",
  "parent_id" int4,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of categories
-- ----------------------------
INSERT INTO "public"."categories" VALUES (4, 'Quần Jean', 'quan-jean', 'Quần jeans các kiểu dáng từ slim fit đến regular', NULL, 1, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (5, 'Quần Short', 'quan-short', 'Quần short thể thao và dạo phố', NULL, 1, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (6, 'Quần Tây', 'quan-tay', 'Quần tây công sở và formal', NULL, 1, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (7, 'Áo Hoodie', 'ao-hoodie', 'Áo hoodie và áo nỉ có mũ', NULL, 3, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (8, 'Áo Blazer', 'ao-blazer', 'Áo blazer và áo vest công sở', NULL, 3, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (9, 'Áo Bomber', 'ao-bomber', 'Áo bomber và áo khoác thể thao', NULL, 3, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (3, 'Áo Khoác', 'ao-khoac', 'Áo khoác, áo vest, blazer', 'http://localhost:3000/category/aokhoac.jpg', NULL, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (2, 'Áo Thun', 'ao-thun', 'Áo thun các loại kiểu dáng', 'http://localhost:3000/category/aothun.jpg', NULL, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');
INSERT INTO "public"."categories" VALUES (1, 'Quần', 'quan', 'Quần các loại - jeans, kaki, short, tây', 'http://localhost:3000/category/quan.jpg', NULL, '2025-11-06 04:46:45.803388', '2025-11-06 04:46:45.803388');

DROP TABLE IF EXISTS "public"."order_items";
CREATE TABLE "public"."order_items" (
  "id" int4 NOT NULL DEFAULT nextval('order_items_id_seq'::regclass),
  "order_id" int4 NOT NULL,
  "product_id" int4,
  "variant_id" int4,
  "product_name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "product_image" varchar(500) COLLATE "pg_catalog"."default",
  "size" varchar(20) COLLATE "pg_catalog"."default",
  "color" varchar(50) COLLATE "pg_catalog"."default",
  "price" numeric(10,2) NOT NULL,
  "quantity" int4 NOT NULL,
  "subtotal" numeric(10,2) NOT NULL,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of order_items
-- ----------------------------
INSERT INTO "public"."order_items" VALUES (1, 1, 5, 9, 'Áo thun', 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 'S', 'Đỏ', 200000.00, 1, 200000.00, '2025-11-08 10:21:04.004409');
INSERT INTO "public"."order_items" VALUES (2, 1, 5, 10, 'Áo thun', 'http://localhost:3000/images/db9d0cd2-d4c4-4190-a215-dc71351f496b.jpg', 'S', 'Xanh Than', 200000.00, 3, 600000.00, '2025-11-08 10:21:04.009609');
INSERT INTO "public"."order_items" VALUES (3, 1, 7, 13, 'Áo khoác kaki', 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 'S', 'Nâu', 310000.00, 1, 310000.00, '2025-11-08 10:21:04.012796');
INSERT INTO "public"."order_items" VALUES (4, 2, 5, 9, 'Áo thun', 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 'S', 'Đỏ', 200000.00, 1, 200000.00, '2025-11-08 10:25:17.727812');
INSERT INTO "public"."order_items" VALUES (5, 2, 5, 10, 'Áo thun', 'http://localhost:3000/images/db9d0cd2-d4c4-4190-a215-dc71351f496b.jpg', 'S', 'Xanh Than', 200000.00, 3, 600000.00, '2025-11-08 10:25:17.731476');
INSERT INTO "public"."order_items" VALUES (6, 2, 7, 13, 'Áo khoác kaki', 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 'S', 'Nâu', 310000.00, 1, 310000.00, '2025-11-08 10:25:17.734067');
INSERT INTO "public"."order_items" VALUES (7, 3, 5, 9, 'Áo thun', 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 'S', 'Đỏ', 200000.00, 1, 200000.00, '2025-11-08 10:25:59.900008');
INSERT INTO "public"."order_items" VALUES (8, 3, 5, 10, 'Áo thun', 'http://localhost:3000/images/db9d0cd2-d4c4-4190-a215-dc71351f496b.jpg', 'S', 'Xanh Than', 200000.00, 3, 600000.00, '2025-11-08 10:25:59.903894');
INSERT INTO "public"."order_items" VALUES (9, 3, 7, 13, 'Áo khoác kaki', 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 'S', 'Nâu', 310000.00, 1, 310000.00, '2025-11-08 10:25:59.906729');
INSERT INTO "public"."order_items" VALUES (10, 4, 5, 9, 'Áo thun', 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 'S', 'Đỏ', 200000.00, 1, 200000.00, '2025-11-08 10:28:24.961784');
INSERT INTO "public"."order_items" VALUES (11, 4, 5, 10, 'Áo thun', 'http://localhost:3000/images/db9d0cd2-d4c4-4190-a215-dc71351f496b.jpg', 'S', 'Xanh Than', 200000.00, 3, 600000.00, '2025-11-08 10:28:24.965859');
INSERT INTO "public"."order_items" VALUES (12, 4, 7, 13, 'Áo khoác kaki', 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 'S', 'Nâu', 310000.00, 1, 310000.00, '2025-11-08 10:28:24.969342');
INSERT INTO "public"."order_items" VALUES (13, 5, 5, 9, 'Áo thun', 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 'S', 'Đỏ', 200000.00, 1, 200000.00, '2025-11-08 10:44:13.603881');
INSERT INTO "public"."order_items" VALUES (14, 5, 5, 10, 'Áo thun', 'http://localhost:3000/images/db9d0cd2-d4c4-4190-a215-dc71351f496b.jpg', 'S', 'Xanh Than', 200000.00, 3, 600000.00, '2025-11-08 10:44:13.608971');
INSERT INTO "public"."order_items" VALUES (15, 6, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:03:38.588199');
INSERT INTO "public"."order_items" VALUES (16, 7, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:04:32.319528');
INSERT INTO "public"."order_items" VALUES (17, 8, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:05:40.26289');
INSERT INTO "public"."order_items" VALUES (18, 9, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:08:55.406124');
INSERT INTO "public"."order_items" VALUES (19, 10, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:11:36.622583');
INSERT INTO "public"."order_items" VALUES (20, 11, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:13:16.556261');
INSERT INTO "public"."order_items" VALUES (21, 12, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:15:21.862483');
INSERT INTO "public"."order_items" VALUES (22, 13, 6, 11, 'Áo khoác Hoodie', 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 'S', 'Xám', 300000.00, 1, 300000.00, '2025-11-09 08:16:01.219643');
INSERT INTO "public"."order_items" VALUES (23, 13, 7, 13, 'Áo khoác kaki', 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 'S', 'Nâu', 310000.00, 1, 310000.00, '2025-11-09 08:16:01.227525');
INSERT INTO "public"."order_items" VALUES (24, 14, 7, 13, 'Áo khoác kaki', 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 'S', 'Nâu', 310000.00, 1, 310000.00, '2025-11-09 08:17:31.628688');

-- ----------------------------
-- Table structure for orders
-- ----------------------------
DROP TABLE IF EXISTS "public"."orders";
CREATE TABLE "public"."orders" (
  "id" int4 NOT NULL DEFAULT nextval('orders_id_seq'::regclass),
  "user_id" int4,
  "order_number" varchar(50) COLLATE "pg_catalog"."default" NOT NULL,
  "full_name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "phone" varchar(20) COLLATE "pg_catalog"."default" NOT NULL,
  "email" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "address" varchar(500) COLLATE "pg_catalog"."default" NOT NULL,
  "ward" varchar(100) COLLATE "pg_catalog"."default" NOT NULL,
  "province" varchar(100) COLLATE "pg_catalog"."default" NOT NULL,
  "status" varchar(20) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'pending'::character varying,
  "payment_method" varchar(20) COLLATE "pg_catalog"."default" NOT NULL DEFAULT 'cod'::character varying,
  "notes" text COLLATE "pg_catalog"."default",
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of orders
-- ----------------------------
INSERT INTO "public"."orders" VALUES (2, 1, 'ORD2511080002', 'xdsd', 'sdsd', 'sdsd', 'sdsdsd', '03472', '12', 'pending', 'cod', NULL, '2025-11-08 10:25:17.717953', '2025-11-08 10:25:17.717953');
INSERT INTO "public"."orders" VALUES (3, 1, 'ORD2511080003', 'xdsd', 'sdsd', 'sdsd', 'sdsdsd', '03472', '12', 'pending', 'cod', NULL, '2025-11-08 10:25:59.888496', '2025-11-08 10:25:59.888496');
INSERT INTO "public"."orders" VALUES (4, 1, 'ORD2511080004', 'xdsd', 'sdsd', 'sdsd', 'sdsdsd', '03472', '12', 'pending', 'cod', NULL, '2025-11-08 10:28:24.956411', '2025-11-08 10:28:24.956411');
INSERT INTO "public"."orders" VALUES (5, 1, 'ORD2511080005', 'sss', 'ssssss', 'sss', 'sss', '28360', '82', 'pending', 'cod', NULL, '2025-11-08 10:44:13.592024', '2025-11-08 10:44:13.592024');
INSERT INTO "public"."orders" VALUES (1, 1, 'ORD2511080001', 'ád', 'ádas', 'áđá', 'ádasd', '20017', '46', 'cancelled', 'cod', NULL, '2025-11-08 10:21:03.995466', '2025-11-08 10:21:03.995466');
INSERT INTO "public"."orders" VALUES (6, 1, 'ORD2511090001', 'sss', 'sss', 'sss', 'sss', '18304', '42', 'pending', 'cod', NULL, '2025-11-09 08:03:38.583499', '2025-11-09 08:03:38.583499');
INSERT INTO "public"."orders" VALUES (7, 1, 'ORD2511090002', 'ss', 'ss', 'ss', 'sss', '30985', '91', 'pending', 'cod', NULL, '2025-11-09 08:04:32.31521', '2025-11-09 08:04:32.31521');
INSERT INTO "public"."orders" VALUES (8, 1, 'ORD2511090003', 'ss', 'ss', 'ss', 'ss', '07654', '24', 'pending', 'cod', NULL, '2025-11-09 08:05:40.251965', '2025-11-09 08:05:40.251965');
INSERT INTO "public"."orders" VALUES (9, 1, 'ORD2511090004', 'ss', 'ss', 'sss', 'ss', '30985', '91', 'pending', 'cod', NULL, '2025-11-09 08:08:55.402637', '2025-11-09 08:08:55.402637');
INSERT INTO "public"."orders" VALUES (10, 1, 'ORD2511090005', 's', 's', 's', 's', '30985', '91', 'pending', 'cod', NULL, '2025-11-09 08:11:36.612379', '2025-11-09 08:11:36.612379');
INSERT INTO "public"."orders" VALUES (11, 1, 'ORD2511090006', 'ss', 'ss', 'ss', 'ss', '07654', '24', 'pending', 'cod', NULL, '2025-11-09 08:13:16.539094', '2025-11-09 08:13:16.539094');
INSERT INTO "public"."orders" VALUES (12, 1, 'ORD2511090007', 'ss', 'ss', 'ss', 'ss', '30985', '91', 'pending', 'cod', NULL, '2025-11-09 08:15:21.842662', '2025-11-09 08:15:21.842662');
INSERT INTO "public"."orders" VALUES (13, 1, 'ORD2511090008', 'áda', 'sda', 'áda', 'ádasd', '18619', '42', 'pending', 'cod', NULL, '2025-11-09 08:16:01.199632', '2025-11-09 08:16:01.199632');
INSERT INTO "public"."orders" VALUES (14, 1, 'ORD2511090009', 'ss', 'ss', 'ss', 'ss', '30985', '91', 'pending', 'cod', NULL, '2025-11-09 08:17:31.608875', '2025-11-09 08:17:31.608875');

-- ----------------------------
-- Table structure for product_images
-- ----------------------------
DROP TABLE IF EXISTS "public"."product_images";
CREATE TABLE "public"."product_images" (
  "id" int4 NOT NULL DEFAULT nextval('product_images_id_seq'::regclass),
  "product_id" int4 NOT NULL,
  "image_url" varchar(500) COLLATE "pg_catalog"."default" NOT NULL,
  "is_primary" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of product_images
-- ----------------------------
INSERT INTO "public"."product_images" VALUES (1, 1, 'http://localhost:3000/images/ce76bb1d-2e6b-4dce-99da-357b2b2b373b.jpg', 't', '2025-11-06 15:34:43.009');
INSERT INTO "public"."product_images" VALUES (6, 6, 'http://localhost:3000/images/8596856d-cbf8-41e9-9809-2aea8fb38628.jpeg', 't', '2025-11-06 17:08:35.527');
INSERT INTO "public"."product_images" VALUES (5, 5, 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 't', '2025-11-06 16:55:35.79');
INSERT INTO "public"."product_images" VALUES (7, 7, 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 't', '2025-11-07 20:06:41.932');

-- ----------------------------
-- Table structure for product_variant_images
-- ----------------------------
DROP TABLE IF EXISTS "public"."product_variant_images";
CREATE TABLE "public"."product_variant_images" (
  "id" int4 NOT NULL DEFAULT nextval('product_variant_images_id_seq'::regclass),
  "variant_id" int4 NOT NULL,
  "image_url" varchar(500) COLLATE "pg_catalog"."default" NOT NULL,
  "is_primary" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of product_variant_images
-- ----------------------------
INSERT INTO "public"."product_variant_images" VALUES (1, 1, 'http://localhost:3000/images/4ec85e7f-6760-4c04-83ee-18fd3bb61fa5.jpg', 't', '2025-11-06 15:34:43.017');
INSERT INTO "public"."product_variant_images" VALUES (2, 2, 'http://localhost:3000/images/8212c8e3-ee0f-475f-a383-cca901d9c5b7.jpg', 't', '2025-11-06 15:34:43.021');
INSERT INTO "public"."product_variant_images" VALUES (10, 10, 'http://localhost:3000/images/db9d0cd2-d4c4-4190-a215-dc71351f496b.jpg', 't', '2025-11-06 16:55:35.801');
INSERT INTO "public"."product_variant_images" VALUES (11, 11, 'http://localhost:3000/images/9c1ba1ab-3ad4-4755-a4ab-e7140927f6eb.jpeg', 't', '2025-11-06 17:08:35.532');
INSERT INTO "public"."product_variant_images" VALUES (12, 12, 'http://localhost:3000/images/77266fcd-4d28-4580-8d81-330b92d66b90.jpeg', 't', '2025-11-06 17:08:35.537');
INSERT INTO "public"."product_variant_images" VALUES (9, 9, 'http://localhost:3000/images/9ab08e72-409f-4063-afee-82a5f4bd8b0d.png', 't', '2025-11-06 16:55:35.797');
INSERT INTO "public"."product_variant_images" VALUES (13, 13, 'http://localhost:3000/images/cd878d84-c944-4d31-b117-80f795abfad5.jpg', 't', '2025-11-07 20:06:41.938');
INSERT INTO "public"."product_variant_images" VALUES (14, 14, 'http://localhost:3000/images/b67b849f-86bf-449d-a58b-56fe7764951b.jpg', 't', '2025-11-07 20:06:41.944');

-- ----------------------------
-- Table structure for product_variants
-- ----------------------------
DROP TABLE IF EXISTS "public"."product_variants";
CREATE TABLE "public"."product_variants" (
  "id" int4 NOT NULL DEFAULT nextval('product_variants_id_seq'::regclass),
  "product_id" int4 NOT NULL,
  "size" varchar(20) COLLATE "pg_catalog"."default",
  "color" varchar(50) COLLATE "pg_catalog"."default",
  "color_code" varchar(20) COLLATE "pg_catalog"."default",
  "stock" int4 NOT NULL DEFAULT 0,
  "price_adjustment" numeric(10,2),
  "sku" varchar(100) COLLATE "pg_catalog"."default",
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of product_variants
-- ----------------------------
INSERT INTO "public"."product_variants" VALUES (1, 1, 'S', 'Đen', '#000000', 20, NULL, 'QUNJEA-S-EN-R2SP', '2025-11-06 15:34:43.013', '2025-11-06 15:34:43.013');
INSERT INTO "public"."product_variants" VALUES (2, 1, 'S', 'Xanh', '#2c6ca3', 20, NULL, 'QUNJEA-S-XAN-M7AG', '2025-11-06 15:34:43.02', '2025-11-06 15:34:43.02');
INSERT INTO "public"."product_variants" VALUES (9, 5, 'S', 'Đỏ', '#d43122', 20, NULL, 'OTHUN-S-05HR', '2025-11-06 16:55:35.794', '2025-11-06 16:55:35.794');
INSERT INTO "public"."product_variants" VALUES (11, 6, 'S', 'Xám', '#818185', 20, NULL, 'OKHOCH-S-XM-RJ8K', '2025-11-06 17:08:35.53', '2025-11-06 17:08:35.53');
INSERT INTO "public"."product_variants" VALUES (12, 6, 'S', 'Đen', '#000000', 20, NULL, 'OKHOCH-S-EN-VSX1', '2025-11-06 17:08:35.535', '2025-11-06 17:08:35.535');
INSERT INTO "public"."product_variants" VALUES (13, 7, 'S', 'Nâu', '#7d5f39', 20, NULL, 'OKHOCK-S-NU-K55A', '2025-11-07 20:06:41.935', '2025-11-07 20:06:41.935');
INSERT INTO "public"."product_variants" VALUES (14, 7, 'S', 'Vàng Nâu', '#d6b387', 20, NULL, 'OKHOCK-S-VNG-NUM3', '2025-11-07 20:06:41.941', '2025-11-07 20:06:41.941');
INSERT INTO "public"."product_variants" VALUES (10, 5, 'S', 'Xanh Than', '#1b285c', 1, NULL, 'OTHUN-S-XAN-CGHT', '2025-11-06 16:55:35.799', '2025-11-06 16:55:35.799');

-- ----------------------------
-- Table structure for products
-- ----------------------------
DROP TABLE IF EXISTS "public"."products";
CREATE TABLE "public"."products" (
  "id" int4 NOT NULL DEFAULT nextval('products_id_seq'::regclass),
  "category_id" int4 NOT NULL,
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "price" numeric(10,2) NOT NULL,
  "sale_price" numeric(10,2),
  "stock" int4 NOT NULL DEFAULT 0,
  "is_featured" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
)
;

-- ----------------------------
-- Records of products
-- ----------------------------
INSERT INTO "public"."products" VALUES (1, 4, 'Quần Jean', 'Quần jean nam', 399000.00, 349000.00, 0, 't', '2025-11-06 15:34:42.978', '2025-11-06 15:34:42.978');
INSERT INTO "public"."products" VALUES (5, 2, 'Áo thun', 'Áo thun nam', 200000.00, 150000.00, 0, 't', '2025-11-06 16:55:35.767', '2025-11-06 16:55:35.767');
INSERT INTO "public"."products" VALUES (6, 7, 'Áo khoác Hoodie', 'Áo khoác Hoodie nam', 300000.00, 250000.00, 0, 't', '2025-11-06 17:08:35.503', '2025-11-06 17:08:35.503');
INSERT INTO "public"."products" VALUES (7, 8, 'Áo khoác kaki', 'Áo khoác kaki', 310000.00, 260000.00, 0, 't', '2025-11-07 20:06:41.912', '2025-11-07 20:06:41.912');
