CREATE TABLE "administrative_units" (
  "id" int4 NOT NULL,
  "full_name" varchar(255) COLLATE "pg_catalog"."default",
  "full_name_en" varchar(255) COLLATE "pg_catalog"."default",
  "short_name" varchar(255) COLLATE "pg_catalog"."default",
  "short_name_en" varchar(255) COLLATE "pg_catalog"."default",
  "code_name" varchar(255) COLLATE "pg_catalog"."default",
  "code_name_en" varchar(255) COLLATE "pg_catalog"."default",
  CONSTRAINT "administrative_units_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "administrative_units" OWNER TO "postgres";

CREATE TABLE "categories" (
  "id" int4 NOT NULL DEFAULT nextval('categories_id_seq'::regclass),
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "slug" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "image" varchar(500) COLLATE "pg_catalog"."default",
  "parent_id" int4,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "categories_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "categories_slug_key" UNIQUE ("slug")
);
ALTER TABLE "categories" OWNER TO "postgres";

CREATE TABLE "chat_messages" (
  "id" int4 NOT NULL DEFAULT nextval('chat_messages_id_seq'::regclass),
  "session_id" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "user_id" int4,
  "sender_type" varchar(20) COLLATE "pg_catalog"."default" NOT NULL,
  "message" text COLLATE "pg_catalog"."default" NOT NULL,
  "is_read" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "chat_messages_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "chat_messages" OWNER TO "postgres";
CREATE INDEX "idx_chat_messages_session_id" ON "chat_messages" USING btree (
  "session_id" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_chat_messages_user_id" ON "chat_messages" USING btree (
  "user_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

CREATE TABLE "kysely_migration" (
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "timestamp" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  CONSTRAINT "kysely_migration_pkey" PRIMARY KEY ("name")
);
ALTER TABLE "kysely_migration" OWNER TO "postgres";

CREATE TABLE "kysely_migration_lock" (
  "id" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "is_locked" int4 NOT NULL DEFAULT 0,
  CONSTRAINT "kysely_migration_lock_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "kysely_migration_lock" OWNER TO "postgres";

CREATE TABLE "order_items" (
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
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "order_items_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "order_items" OWNER TO "postgres";

CREATE TABLE "orders" (
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
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "orders_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "orders_order_number_key" UNIQUE ("order_number")
);
ALTER TABLE "orders" OWNER TO "postgres";
CREATE INDEX "idx_orders_status" ON "orders" USING btree (
  "status" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_orders_user" ON "orders" USING btree (
  "user_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

CREATE TABLE "product_images" (
  "id" int4 NOT NULL DEFAULT nextval('product_images_id_seq'::regclass),
  "product_id" int4 NOT NULL,
  "image_url" varchar(500) COLLATE "pg_catalog"."default" NOT NULL,
  "is_primary" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "product_images_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "product_images" OWNER TO "postgres";

CREATE TABLE "product_variant_images" (
  "id" int4 NOT NULL DEFAULT nextval('product_variant_images_id_seq'::regclass),
  "variant_id" int4 NOT NULL,
  "image_url" varchar(500) COLLATE "pg_catalog"."default" NOT NULL,
  "is_primary" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "product_variant_images_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "product_variant_images" OWNER TO "postgres";

CREATE TABLE "product_variants" (
  "id" int4 NOT NULL DEFAULT nextval('product_variants_id_seq'::regclass),
  "product_id" int4 NOT NULL,
  "size" varchar(20) COLLATE "pg_catalog"."default",
  "color" varchar(50) COLLATE "pg_catalog"."default",
  "color_code" varchar(20) COLLATE "pg_catalog"."default",
  "stock" int4 NOT NULL DEFAULT 0,
  "price_adjustment" numeric(10,2),
  "sku" varchar(100) COLLATE "pg_catalog"."default",
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "product_variants_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "product_variants" OWNER TO "postgres";

CREATE TABLE "products" (
  "id" int4 NOT NULL DEFAULT nextval('products_id_seq'::regclass),
  "category_id" int4 NOT NULL,
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "description" text COLLATE "pg_catalog"."default",
  "price" numeric(10,2) NOT NULL,
  "sale_price" numeric(10,2),
  "stock" int4 NOT NULL DEFAULT 0,
  "is_featured" bool NOT NULL DEFAULT false,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "products_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "products" OWNER TO "postgres";
CREATE INDEX "idx_products_category" ON "products" USING btree (
  "category_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

CREATE TABLE "provinces" (
  "code" varchar(20) COLLATE "pg_catalog"."default" NOT NULL,
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "name_en" varchar(255) COLLATE "pg_catalog"."default",
  "full_name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "full_name_en" varchar(255) COLLATE "pg_catalog"."default",
  "code_name" varchar(255) COLLATE "pg_catalog"."default",
  "administrative_unit_id" int4,
  CONSTRAINT "provinces_pkey" PRIMARY KEY ("code")
);
ALTER TABLE "provinces" OWNER TO "postgres";
CREATE INDEX "idx_provinces_unit" ON "provinces" USING btree (
  "administrative_unit_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

CREATE TABLE "seed_history" (
  "id" int4 NOT NULL DEFAULT nextval('seed_history_id_seq'::regclass),
  "seed_name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "executed_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "seed_history_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "seed_history_seed_name_key" UNIQUE ("seed_name")
);
ALTER TABLE "seed_history" OWNER TO "postgres";

CREATE TABLE "shopping_cart" (
  "id" int4 NOT NULL DEFAULT nextval('shopping_cart_id_seq'::regclass),
  "user_id" int4 NOT NULL,
  "product_id" int4 NOT NULL,
  "variant_id" int4,
  "quantity" int4 NOT NULL,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "shopping_cart_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "shopping_cart" OWNER TO "postgres";

CREATE TABLE "users" (
  "id" int4 NOT NULL DEFAULT nextval('users_id_seq'::regclass),
  "email" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "password" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "full_name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "phone" varchar(20) COLLATE "pg_catalog"."default",
  "avatar" varchar(500) COLLATE "pg_catalog"."default",
  "address" varchar(500) COLLATE "pg_catalog"."default",
  "ward" varchar(100) COLLATE "pg_catalog"."default",
  "province" varchar(100) COLLATE "pg_catalog"."default",
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "users_pkey" PRIMARY KEY ("id"),
  CONSTRAINT "users_email_key" UNIQUE ("email")
);
ALTER TABLE "users" OWNER TO "postgres";

CREATE TABLE "wards" (
  "code" varchar(20) COLLATE "pg_catalog"."default" NOT NULL,
  "name" varchar(255) COLLATE "pg_catalog"."default" NOT NULL,
  "name_en" varchar(255) COLLATE "pg_catalog"."default",
  "full_name" varchar(255) COLLATE "pg_catalog"."default",
  "full_name_en" varchar(255) COLLATE "pg_catalog"."default",
  "code_name" varchar(255) COLLATE "pg_catalog"."default",
  "province_code" varchar(20) COLLATE "pg_catalog"."default",
  "administrative_unit_id" int4,
  CONSTRAINT "wards_pkey" PRIMARY KEY ("code")
);
ALTER TABLE "wards" OWNER TO "postgres";
CREATE INDEX "idx_wards_province" ON "wards" USING btree (
  "province_code" COLLATE "pg_catalog"."default" "pg_catalog"."text_ops" ASC NULLS LAST
);
CREATE INDEX "idx_wards_unit" ON "wards" USING btree (
  "administrative_unit_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

CREATE TABLE "wishlist" (
  "id" int4 NOT NULL DEFAULT nextval('wishlist_id_seq'::regclass),
  "user_id" int4 NOT NULL,
  "product_id" int4 NOT NULL,
  "created_at" timestamp(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "wishlist_pkey" PRIMARY KEY ("id")
);
ALTER TABLE "wishlist" OWNER TO "postgres";
CREATE UNIQUE INDEX "wishlist_user_product_unique" ON "wishlist" USING btree (
  "user_id" "pg_catalog"."int4_ops" ASC NULLS LAST,
  "product_id" "pg_catalog"."int4_ops" ASC NULLS LAST
);

ALTER TABLE "categories" ADD CONSTRAINT "categories_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "categories" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "chat_messages" ADD CONSTRAINT "chat_messages_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "order_items" ADD CONSTRAINT "order_items_order_id_fkey" FOREIGN KEY ("order_id") REFERENCES "orders" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "order_items" ADD CONSTRAINT "order_items_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "products" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "order_items" ADD CONSTRAINT "order_items_variant_id_fkey" FOREIGN KEY ("variant_id") REFERENCES "product_variants" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "orders" ADD CONSTRAINT "orders_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "product_images" ADD CONSTRAINT "product_images_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "products" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "product_variant_images" ADD CONSTRAINT "product_variant_images_variant_id_fkey" FOREIGN KEY ("variant_id") REFERENCES "product_variants" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "product_variants" ADD CONSTRAINT "product_variants_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "products" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "products" ADD CONSTRAINT "products_category_id_fkey" FOREIGN KEY ("category_id") REFERENCES "categories" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "provinces" ADD CONSTRAINT "provinces_administrative_unit_id_fkey" FOREIGN KEY ("administrative_unit_id") REFERENCES "administrative_units" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "shopping_cart" ADD CONSTRAINT "shopping_cart_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "products" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "shopping_cart" ADD CONSTRAINT "shopping_cart_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "shopping_cart" ADD CONSTRAINT "shopping_cart_variant_id_fkey" FOREIGN KEY ("variant_id") REFERENCES "product_variants" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "wards" ADD CONSTRAINT "wards_administrative_unit_id_fkey" FOREIGN KEY ("administrative_unit_id") REFERENCES "administrative_units" ("id") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "wards" ADD CONSTRAINT "wards_province_code_fkey" FOREIGN KEY ("province_code") REFERENCES "provinces" ("code") ON DELETE SET NULL ON UPDATE NO ACTION;
ALTER TABLE "wishlist" ADD CONSTRAINT "wishlist_product_id_fkey" FOREIGN KEY ("product_id") REFERENCES "products" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;
ALTER TABLE "wishlist" ADD CONSTRAINT "wishlist_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE NO ACTION;

