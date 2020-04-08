--accepted
WITH failedreads as
(SELECT meter_id, CHANNEL_ID, read_dtm FROM REGISTERREADS_HISTORY ih WHERE READ_VERSION = 1 AND READ_DTM >= '01-JAN-2019' AND STATUS = 2
UNION
SELECT meter_id, CHANNEL_ID, read_dtm FROM REGISTERREADS r WHERE READ_DTM >= '01-JAN-2019' AND STATUS = 2
),
currentread as
(SELECT i.meter_id, i.CHANNEL_ID, i.read_dtm, i.status, i.read_version, i.read_value, rrc.COMMENTS FROM registerreads i, failedreads f, REGISTERREADCOMMENTS rrc
WHERE i.meter_id = f.meter_id
  AND i.CHANNEL_ID = f.channel_id
  AND i.read_dtm = f.read_dtm
  AND rrc.READ_DTM = i.READ_DTM 
  AND rrc.CHANNEL_ID  = i.CHANNEL_ID 
  AND rrc.METER_ID = i.METER_ID 
  AND rrc.comments NOT LIKE '%Automatically%'),
acceptedreads as
(SELECT r.meter_id, r.channel_id, r.read_dtm, r.status, r.read_version, r.read_value, r.comments 
FROM currentread r 
WHERE r.status = 5 
AND r.comments LIKE '%Accepted%'),
prevread as
(SELECT meter_id, channel_id, read_dtm, read_value, comments, 
(SELECT max(read_dtm) FROM REGISTERREADS r WHERE r.METER_ID = a.meter_id AND r.CHANNEL_ID = a.channel_id AND r.READ_DTM < a.read_dtm) AS prev_read_dtm
FROM acceptedreads a),
prevprevread AS
(SELECT meter_id, channel_id, READ_dtm, read_value, comments, prev_read_dtm, 
(SELECT max(read_dtm) FROM REGISTERREADS r WHERE r.METER_ID = a.meter_id AND r.CHANNEL_ID = a.channel_id AND r.READ_DTM < a.prev_read_dtm) AS prev_read_dtm2
FROM prevread a)
SELECT pr.meter_id, l.LOCATION_NO, l.LOCATION_CLASS, l.BILLING_CYCLE, to_char(pr.read_dtm, 'YYYY') AS "YEAR", to_char(pr.read_dtm, 'MM') AS "MONTH", to_char(pr.READ_dtm, 'DD') AS "DAY", 
to_char(pr.read_dtm, 'DY') AS "DOW", pr.comments,
pr.read_value, r.read_value AS prev_read, r2.read_value AS prev_read2, to_char(pr.prev_read_dtm, 'YYYY') AS "PREVYEAR", to_char(pr.prev_read_dtm, 'MM') AS "PREVMONTH", to_char(pr.prev_read_dtm, 'DD') AS "PREVDAY", 
to_char(pr.prev_read_dtm, 'DY') AS "PREVDOW", to_char(pr.prev_read_dtm2, 'YYYY') AS "PREVPREVYEAR", to_char(pr.prev_read_dtm2, 'MM') AS "PREVPREVMONTH", to_char(pr.prev_read_dtm2, 'DD') AS "PREVPREVDAY", 
to_char(pr.prev_read_dtm2, 'DY') AS "PREVPREVDOW",r.uom, r.status AS prev_status, r2.status AS prev_status2,
round((SELECT min(b.bill_dt) - pr.read_dtm FROM billing_schedule b WHERE l.BILLING_CYCLE = b.BILLING_CYCLE AND trunc(pr.read_dtm) BETWEEN b.PERIOD_START_DT AND b.PERIOD_END_DT)) AS days_from_billdt,
CASE WHEN instr(comments, 'HiLo:') >= 1 THEN 1 ELSE 0 END AS HiLo,
CASE WHEN instr(comments, 'RegisterIncreasing:') >= 1 THEN 1 ELSE 0 END AS RegisterIncreasing,
CASE WHEN instr(comments, 'SumCheck:') >= 1 THEN 1 ELSE 0 END AS SumCheck,
CASE WHEN instr(comments, 'MaxDailyUsage:') >= 1 THEN 1 ELSE 0 END AS MaxDailyUsage
FROM prevprevread pr, registerreads r, registerreads r2, meter_location_xref mlx, locations l 
WHERE pr.meter_id = r.METER_ID 
  AND pr.channel_id = r.channel_id
  AND pr.prev_read_dtm = r.read_dtm
  AND pr.meter_id = r2.METER_ID 
  AND pr.channel_id = r2.channel_id
  AND pr.prev_read_dtm2 = r2.read_dtm
  AND pr.meter_id = mlx.METER_ID 
  AND mlx.ACTIVE_DT <= pr.read_dtm
  AND mlx.INACTIVE_DT > pr.read_dtm
  AND l.location_no = mlx.LOCATION_NO 
ORDER BY pr.read_dtm
;

--failed
WITH failedreads as
(SELECT meter_id, CHANNEL_ID, read_dtm FROM REGISTERREADS_HISTORY ih WHERE READ_VERSION = 1 AND READ_DTM >= '01-JAN-2019' AND STATUS = 2
UNION
SELECT meter_id, CHANNEL_ID, read_dtm FROM REGISTERREADS r WHERE READ_DTM >= '01-JAN-2019' AND STATUS = 2
),
currentread as
(SELECT i.meter_id, i.CHANNEL_ID, i.read_dtm, i.status, i.read_version, i.read_value, rrc.COMMENTS FROM registerreads i, failedreads f, REGISTERREADCOMMENTS rrc
WHERE i.meter_id = f.meter_id
  AND i.CHANNEL_ID = f.channel_id
  AND i.read_dtm = f.read_dtm
  AND rrc.READ_DTM = i.READ_DTM 
  AND rrc.CHANNEL_ID  = i.CHANNEL_ID 
  AND rrc.METER_ID = i.METER_ID 
  AND rrc.comments NOT LIKE '%Automatically%'),
acceptedreads as
(SELECT r.meter_id, r.channel_id, r.read_dtm, r.status, r.read_version, r.read_value, r.comments 
FROM currentread r 
WHERE r.status = 2),
prevread as
(SELECT meter_id, channel_id, read_dtm, read_value, comments, 
(SELECT max(read_dtm) FROM REGISTERREADS r WHERE r.METER_ID = a.meter_id AND r.CHANNEL_ID = a.channel_id AND r.READ_DTM < a.read_dtm) AS prev_read_dtm
FROM acceptedreads a),
prevprevread AS
(SELECT meter_id, channel_id, READ_dtm, read_value, comments, prev_read_dtm, 
(SELECT max(read_dtm) FROM REGISTERREADS r WHERE r.METER_ID = a.meter_id AND r.CHANNEL_ID = a.channel_id AND r.READ_DTM < a.prev_read_dtm) AS prev_read_dtm2
FROM prevread a)
SELECT pr.meter_id, l.LOCATION_NO, l.LOCATION_CLASS, l.BILLING_CYCLE, to_char(pr.read_dtm, 'YYYY') AS "YEAR", to_char(pr.read_dtm, 'MM') AS "MONTH", to_char(pr.READ_dtm, 'DD') AS "DAY", 
to_char(pr.read_dtm, 'DY') AS "DOW", pr.comments,
pr.read_value, r.read_value AS prev_read, r2.read_value AS prev_read2, to_char(pr.prev_read_dtm, 'YYYY') AS "PREVYEAR", to_char(pr.prev_read_dtm, 'MM') AS "PREVMONTH", to_char(pr.prev_read_dtm, 'DD') AS "PREVDAY", 
to_char(pr.prev_read_dtm, 'DY') AS "PREVDOW", to_char(pr.prev_read_dtm2, 'YYYY') AS "PREVPREVYEAR", to_char(pr.prev_read_dtm2, 'MM') AS "PREVPREVMONTH", to_char(pr.prev_read_dtm2, 'DD') AS "PREVPREVDAY", 
to_char(pr.prev_read_dtm2, 'DY') AS "PREVPREVDOW",r.uom, r.status AS prev_status, r2.status AS prev_status2,
round((SELECT min(b.bill_dt) - pr.read_dtm FROM billing_schedule b WHERE l.BILLING_CYCLE = b.BILLING_CYCLE AND trunc(pr.read_dtm) BETWEEN b.PERIOD_START_DT AND b.PERIOD_END_DT)) AS days_from_billdt,
CASE WHEN instr(comments, 'HiLo:') >= 1 THEN 1 ELSE 0 END AS HiLo,
CASE WHEN instr(comments, 'RegisterIncreasing:') >= 1 THEN 1 ELSE 0 END AS RegisterIncreasing,
CASE WHEN instr(comments, 'SumCheck:') >= 1 THEN 1 ELSE 0 END AS SumCheck,
CASE WHEN instr(comments, 'MaxDailyUsage:') >= 1 THEN 1 ELSE 0 END AS MaxDailyUsage
FROM prevprevread pr, registerreads r, registerreads r2, meter_location_xref mlx, locations l 
WHERE pr.meter_id = r.METER_ID 
  AND pr.channel_id = r.channel_id
  AND pr.prev_read_dtm = r.read_dtm
  AND pr.meter_id = r2.METER_ID 
  AND pr.channel_id = r2.channel_id
  AND pr.prev_read_dtm2 = r2.read_dtm
  AND pr.meter_id = mlx.METER_ID 
  AND mlx.ACTIVE_DT <= pr.read_dtm
  AND mlx.INACTIVE_DT > pr.read_dtm
  AND l.location_no = mlx.LOCATION_NO 
ORDER BY pr.read_dtm;
