-- Latent Raw evidence tables (Postgres). Zetaris reads these; the agent reads them
-- only through run_sql. Every row has an evidence_id the decision log cites.
-- Run one statement at a time if you paste this into Zetaris.

CREATE TABLE IF NOT EXISTS shots (
    shot_id        text PRIMARY KEY,           -- e.g. 'shot01'
    title          text,
    source_model   text,                       -- Seedance / Veo / Kling
    source_file    text,                       -- 8-bit MP4 name (our own footage only)
    fps            numeric,
    frame_count    int,
    notes          text
);

CREATE TABLE IF NOT EXISTS exr_headers (
    evidence_id                 text PRIMARY KEY,   -- 'hdr:shot01:f120'
    shot_id                     text REFERENCES shots,
    frame                       int,
    width                       int,
    height                      int,
    compression                 text,
    chromaticities              text,
    aces_image_container_flag   text,
    header_json                 jsonb
);

CREATE TABLE IF NOT EXISTS frame_range (
    evidence_id       text PRIMARY KEY,             -- 'range:shot01:f120'
    shot_id           text REFERENCES shots,
    frame             int,
    src_clipped_pct   numeric,   -- % of 8-bit pixels with a channel at 250+
    src_peak_lum      numeric,   -- linear Rec.709, 0..1
    hdr_peak_lum      numeric,   -- linear ACES
    hdr_mean_lum      numeric,
    hdr_range_stops   numeric,   -- log2(p99.9 / p0.1) of the EXR
    synthesized_pct   numeric    -- % of the frame the model synthesized
);

CREATE TABLE IF NOT EXISTS item_metrics (
    evidence_id       text PRIMARY KEY,             -- 'item:shot01:f120:sky'
    shot_id           text REFERENCES shots,
    frame             int,
    item              text,      -- SAM 3.1 label: car, sky, buildings
    area_pct          numeric,
    clipped_pct       numeric,   -- 8-bit source, channel at 250+
    mean_lum          numeric,   -- 8-bit source, linear Rec.709
    peak_lum          numeric,
    mean_saturation   numeric,
    r_over_g          numeric,   -- colour balance, 1.0 = neutral
    b_over_g          numeric,
    synthesized_pct   numeric,
    hdr_peak_lum      numeric,   -- EXR, linear ACES
    hdr_mean_lum      numeric,
    hdr_r_over_g      numeric,
    hdr_b_over_g      numeric
);

CREATE TABLE IF NOT EXISTS balance_checks (
    evidence_id     text PRIMARY KEY,   -- 'balance:shot01:shot02:car:after'
    shot_a          text REFERENCES shots,
    shot_b          text REFERENCES shots,
    frame_a         int,
    frame_b         int,
    item            text,
    stage           text CHECK (stage IN ('before', 'after')),
    lum_gap_stops   numeric,   -- log2(b / a) of mean luminance
    r_over_g_gap    numeric,
    b_over_g_gap    numeric,
    cites           jsonb      -- the item_metrics evidence_ids compared
);

CREATE TABLE IF NOT EXISTS vision_checks (
    evidence_id   text PRIMARY KEY,                 -- 'vision:shot01:f120:contents'
    shot_id       text REFERENCES shots,
    frame         int,
    question      text,
    answer        text,
    job_id        text,
    model         text,
    receipt       jsonb,          -- meterless.receipt from Nightshift
    created_at    timestamptz DEFAULT now()
);

-- One place for the agent to start: every citable fact for a shot.
CREATE OR REPLACE VIEW shot_evidence AS
SELECT i.shot_id, i.frame, i.evidence_id, 'item' AS kind, i.item AS subject,
       jsonb_build_object('clipped_pct', i.clipped_pct, 'mean_lum', i.mean_lum,
                          'peak_lum', i.peak_lum, 'mean_saturation', i.mean_saturation,
                          'r_over_g', i.r_over_g, 'b_over_g', i.b_over_g,
                          'synthesized_pct', i.synthesized_pct, 'hdr_peak_lum', i.hdr_peak_lum) AS facts
FROM item_metrics i
UNION ALL
SELECT f.shot_id, f.frame, f.evidence_id, 'range', 'frame',
       jsonb_build_object('src_clipped_pct', f.src_clipped_pct, 'hdr_range_stops', f.hdr_range_stops,
                          'hdr_peak_lum', f.hdr_peak_lum, 'synthesized_pct', f.synthesized_pct)
FROM frame_range f
UNION ALL
SELECT b.shot_a, b.frame_a, b.evidence_id, 'balance', b.item,
       jsonb_build_object('vs', b.shot_b, 'stage', b.stage, 'lum_gap_stops', b.lum_gap_stops,
                          'r_over_g_gap', b.r_over_g_gap, 'b_over_g_gap', b.b_over_g_gap)
FROM balance_checks b
UNION ALL
SELECT v.shot_id, v.frame, v.evidence_id, 'vision', v.question,
       jsonb_build_object('answer', v.answer, 'model', v.model)
FROM vision_checks v;
