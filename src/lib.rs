use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

pub mod elements;


/// Spatial address for a times-plank output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Address {
    pub x: i64,
    pub y: i64,
    pub z: i64,
}

/// A single addressed output produced by an agent/transformation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlankOutput {
    pub value: i64,
    pub address: Address,
    pub agent_id: usize,
    pub hash: u64,
}

/// One grid row with a base hash and addressed outputs.
#[derive(Debug, Clone, Default)]
pub struct GridRow {
    pub base_hash: u64,
    pub cells: Vec<PlankOutput>,
}

/// Times-plank grid collecting multi-output multiplication results.
#[derive(Debug, Clone, Default)]
pub struct TimesPlankGrid {
    pub rows: Vec<GridRow>,
    pub log: Vec<String>,
    pub num_positive: usize,
    pub num_negative: usize,
}

/// Timing information for a calculation run.
#[derive(Debug, Clone, Default)]
pub struct CalcTiming {
    pub per_row: Vec<Duration>,
    pub total: Duration,
}

/// A timed grid run (e.g. year/x series calculations).
#[derive(Debug, Clone, Default)]
pub struct TimedRun {
    pub grid: TimesPlankGrid,
    pub timing: CalcTiming,
}

/// Supported math operations for times-plank / tree events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MathOp {
    Mul,
    Div,
}

/// Conventional reference multiplication.
pub fn standard_multiply(a: i64, b: i64) -> i64 {
    a * b
}

/// Conventional reference integer division (0 when divisor is 0).
pub fn standard_divide(a: i64, b: i64) -> i64 {
    if b == 0 { 0 } else { a / b }
}

/// Convert a number into digit addresses (most-significant first).
/// Address x = digit position (0-based from left), y = digit value, z = 0.
pub fn digit_addresses(n: i64) -> Vec<Address> {
    let s = n.abs().to_string();
    s.chars()
        .enumerate()
        .map(|(i, c)| Address {
            x: i as i64,
            y: c.to_digit(10).unwrap_or(0) as i64,
            z: 0,
        })
        .collect()
}

/// Compute a stable hash/score for an output candidate.
pub fn score_hash(a: i64, b: i64, agent_id: usize, value: i64, address: Address) -> u64 {
    score_hash_op(MathOp::Mul, a, b, agent_id, value, address)
}

/// Compute a stable hash/score for an output candidate including operation.
pub fn score_hash_op(op: MathOp, a: i64, b: i64, agent_id: usize, value: i64, address: Address) -> u64 {
    let mut hasher = DefaultHasher::new();
    op.hash(&mut hasher);
    a.hash(&mut hasher);
    b.hash(&mut hasher);
    agent_id.hash(&mut hasher);
    value.hash(&mut hasher);
    address.hash(&mut hasher);
    hasher.finish()
}

/// Transform addressed digits by multiplier and reconstruct a primary value.
/// Rule: multiply each digit independently and concatenate the digit-products.
/// Example: 12*8 -> digits (1,2) -> (8,16) -> 816.
fn primary_digit_transform(a: i64, b: i64) -> Option<i64> {
    let digits = digit_addresses(a);
    if digits.is_empty() {
        return None;
    }
    let mut s = String::new();
    for d in digits {
        let p = d.y * b;
        s.push_str(&p.to_string());
    }
    s.parse::<i64>().ok().map(|v| if a < 0 { -v } else { v })
}

/// Transform addressed digits by divisor and reconstruct a primary spatial value.
/// Rule: divide each digit independently (integer division) and concatenate results.
/// Example: 84/4 with digits (8,4) -> (2,1) -> 21. Division by 0 yields None.
fn primary_digit_div_transform(a: i64, b: i64) -> Option<i64> {
    if b == 0 {
        return None;
    }
    let digits = digit_addresses(a);
    if digits.is_empty() {
        return None;
    }
    let mut s = String::new();
    for d in digits {
        let q = d.y / b;
        s.push_str(&q.abs().to_string());
    }
    if s.is_empty() {
        s.push('0');
    }
    s.parse::<i64>().ok().map(|v| {
        let sign_neg = (a < 0) ^ (b < 0);
        if sign_neg { -v } else { v }
    })
}

/// Generate proposed spatial outputs for a*b (up to multiple per standard result).
/// Agent 0: primary per-digit address transform.
/// Agent 1: standard arithmetic result (reference).
/// Agent 2+: alternative spatial compositions derived from digit products.
pub fn times_plank_multiply(a: i64, b: i64) -> Vec<PlankOutput> {
    let mut outputs = Vec::new();
    let std = standard_multiply(a, b);

    // Agent 0: primary digit transform
    if let Some(v) = primary_digit_transform(a, b) {
        let addr = Address { x: 0, y: 0, z: 0 };
        outputs.push(PlankOutput {
            value: v,
            address: addr,
            agent_id: 0,
            hash: score_hash_op(MathOp::Mul, a, b, 0, v, addr),
        });
    }

    // Agent 1: standard result
    let addr_std = Address { x: 1, y: 0, z: 0 };
    outputs.push(PlankOutput {
        value: std,
        address: addr_std,
        agent_id: 1,
        hash: score_hash_op(MathOp::Mul, a, b, 1, std, addr_std),
    });

    // Agent 2/3: alternative compositions from per-digit products (spatial variants)
    let digits = digit_addresses(a);
    if digits.len() >= 2 {
        let products: Vec<i64> = digits.iter().map(|d| d.y * b).collect();
        // Variant A: first product + sum of remaining products magnitude
        let sum_rest: i64 = products.iter().skip(1).sum();
        let v_a_str = format!("{}{}", products[0].abs(), sum_rest.abs());
        if let Ok(v_a) = v_a_str.parse::<i64>() {
            let addr = Address { x: 2, y: 0, z: 1 };
            let value = if a < 0 { -v_a } else { v_a };
            outputs.push(PlankOutput {
                value,
                address: addr,
                agent_id: 2,
                hash: score_hash_op(MathOp::Mul, a, b, 2, value, addr),
            });
        }
        // Variant B: (first + second) blended with last product digit
        let head_sum: i64 = products[0] + products.get(1).copied().unwrap_or(0);
        let tail = products.last().copied().unwrap_or(0);
        let v_b_str = format!("{}{}", head_sum.abs(), tail.abs() % 10);
        if let Ok(v_b) = v_b_str.parse::<i64>() {
            let addr = Address { x: 3, y: 0, z: 2 };
            let value = if a < 0 { -v_b } else { v_b };
            outputs.push(PlankOutput {
                value,
                address: addr,
                agent_id: 3,
                hash: score_hash_op(MathOp::Mul, a, b, 3, value, addr),
            });
        }
    }

    outputs
}

/// Generate proposed spatial outputs for a/b (division table/times-plank division).
/// Agent 0: primary per-digit spatial division transform.
/// Agent 1: standard integer division result (reference, 0 if b==0).
/// Agent 2+: alternative compositions from per-digit quotients/remainders.
pub fn times_plank_divide(a: i64, b: i64) -> Vec<PlankOutput> {
    let mut outputs = Vec::new();
    let std = standard_divide(a, b);

    if let Some(v) = primary_digit_div_transform(a, b) {
        let addr = Address { x: 0, y: 0, z: 10 };
        outputs.push(PlankOutput {
            value: v,
            address: addr,
            agent_id: 0,
            hash: score_hash_op(MathOp::Div, a, b, 0, v, addr),
        });
    }

    let addr_std = Address { x: 1, y: 0, z: 10 };
    outputs.push(PlankOutput {
        value: std,
        address: addr_std,
        agent_id: 1,
        hash: score_hash_op(MathOp::Div, a, b, 1, std, addr_std),
    });

    if b != 0 {
        let digits = digit_addresses(a);
        if digits.len() >= 2 {
            let quots: Vec<i64> = digits.iter().map(|d| d.y / b).collect();
            let rems: Vec<i64> = digits.iter().map(|d| d.y % b).collect();
            // Variant A: concatenated per-digit quotients + remainder sum marker
            let sum_rem: i64 = rems.iter().sum();
            let mut v_a_str = String::new();
            for q in &quots {
                v_a_str.push_str(&q.abs().to_string());
            }
            v_a_str.push_str(&sum_rem.abs().to_string());
            if let Ok(v_a) = v_a_str.parse::<i64>() {
                let addr = Address { x: 2, y: 0, z: 11 };
                let sign_neg = (a < 0) ^ (b < 0);
                let value = if sign_neg { -v_a } else { v_a };
                outputs.push(PlankOutput {
                    value,
                    address: addr,
                    agent_id: 2,
                    hash: score_hash_op(MathOp::Div, a, b, 2, value, addr),
                });
            }
            // Variant B: standard quotient and remainder encoded as addressed pair value
            let rem_total = a % b;
            let v_b_str = format!("{}{}", std.abs(), rem_total.abs());
            if let Ok(v_b) = v_b_str.parse::<i64>() {
                let addr = Address { x: 3, y: 0, z: 12 };
                let sign_neg = (a < 0) ^ (b < 0);
                let value = if sign_neg { -v_b } else { v_b };
                outputs.push(PlankOutput {
                    value,
                    address: addr,
                    agent_id: 3,
                    hash: score_hash_op(MathOp::Div, a, b, 3, value, addr),
                });
            }
        } else {
            // single digit: include remainder-encoded variant
            let rem_total = a % b;
            let v_b_str = format!("{}{}", std.abs(), rem_total.abs());
            if let Ok(v_b) = v_b_str.parse::<i64>() {
                let addr = Address { x: 3, y: 0, z: 12 };
                let sign_neg = (a < 0) ^ (b < 0);
                let value = if sign_neg { -v_b } else { v_b };
                outputs.push(PlankOutput {
                    value,
                    address: addr,
                    agent_id: 3,
                    hash: score_hash_op(MathOp::Div, a, b, 3, value, addr),
                });
            }
        }
    }

    outputs
}

/// Agent sorting: sort outputs by score/hash (ascending), stable by agent_id.
pub fn sort_agents_by_score(outputs: &mut [PlankOutput]) {
    outputs.sort_by(|a, b| a.hash.cmp(&b.hash).then(a.agent_id.cmp(&b.agent_id)));
}

/// Build a grid row for an operation, with base hash derived from (op, inputs).
pub fn build_row_op(op: MathOp, a: i64, b: i64) -> GridRow {
    let mut outputs = match op {
        MathOp::Mul => times_plank_multiply(a, b),
        MathOp::Div => times_plank_divide(a, b),
    };
    sort_agents_by_score(&mut outputs);
    let mut hasher = DefaultHasher::new();
    op.hash(&mut hasher);
    a.hash(&mut hasher);
    b.hash(&mut hasher);
    GridRow {
        base_hash: hasher.finish(),
        cells: outputs,
    }
}

/// Build a grid row for a multiplication (compat).
pub fn build_row(a: i64, b: i64) -> GridRow {
    build_row_op(MathOp::Mul, a, b)
}

/// Build a grid for a set of (op, a, b) calculations.
pub fn build_grid_ops(calcs: &[(MathOp, i64, i64)]) -> TimesPlankGrid {
    let mut grid = TimesPlankGrid::default();
    for &(op, a, b) in calcs {
        let row = build_row_op(op, a, b);
        for cell in &row.cells {
            if cell.value >= 0 {
                grid.num_positive += 1;
            } else {
                grid.num_negative += 1;
            }
        }
        let sym = match op { MathOp::Mul => "x", MathOp::Div => "/" };
        grid.log.push(format!(
            "row base_hash={} outs={} for {}{}{}",
            row.base_hash, row.cells.len(), a, sym, b
        ));
        grid.rows.push(row);
    }
    grid
}

/// Build a grid for a set of multiplication pairs (compat).
pub fn build_grid(pairs: &[(i64, i64)]) -> TimesPlankGrid {
    let calcs: Vec<(MathOp, i64, i64)> = pairs.iter().map(|&(a, b)| (MathOp::Mul, a, b)).collect();
    build_grid_ops(&calcs)
}

/// Generate a multiplication/times table: base * 1..=n
pub fn times_table(base: i64, n: i64) -> TimesPlankGrid {
    let pairs: Vec<(i64, i64)> = (1..=n.max(0)).map(|i| (base, i)).collect();
    build_grid(&pairs)
}

/// Generate a division table: dividend / 1..=n (skips 0)
pub fn division_table(dividend: i64, n: i64) -> TimesPlankGrid {
    let calcs: Vec<(MathOp, i64, i64)> = (1..=n.max(0)).map(|i| (MathOp::Div, dividend, i)).collect();
    build_grid_ops(&calcs)
}

/// Run calculations over a series using a timer for the given operation.
/// Each x is represented as (base op x), forming the timeline.
pub fn run_series_op(op: MathOp, base: i64, xs: &[i64]) -> TimedRun {
    let mut grid = TimesPlankGrid::default();
    let mut timing = CalcTiming::default();
    let total_start = Instant::now();
    let sym = match op { MathOp::Mul => "x", MathOp::Div => "/" };

    for &x in xs {
        let row_start = Instant::now();
        let row = build_row_op(op, base, x);
        let elapsed = row_start.elapsed();
        timing.per_row.push(elapsed);

        for cell in &row.cells {
            if cell.value >= 0 {
                grid.num_positive += 1;
            } else {
                grid.num_negative += 1;
            }
        }
        grid.log.push(format!(
            "series={}{} base_hash={} outs={} elapsed_ns={}",
            x, sym, row.base_hash, row.cells.len(), elapsed.as_nanos()
        ));
        grid.rows.push(row);
    }

    timing.total = total_start.elapsed();
    TimedRun { grid, timing }
}

/// Run calculations over a year/x multiplication series using a timer (compat).
/// Example years: [1, 2, ..., 1000, 8000].
pub fn run_year_series(base: i64, years: &[i64]) -> TimedRun {
    run_series_op(MathOp::Mul, base, years)
}

/// Run calculations over a division series using a timer.
pub fn run_division_series(dividend: i64, divisors: &[i64]) -> TimedRun {
    run_series_op(MathOp::Div, dividend, divisors)
}

/// Base clock speed (outputs per second) derived from timed timeline run.
pub fn base_clock_outputs_per_sec(run: &TimedRun) -> f64 {
    let secs = run.timing.total.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    await_outs(&run.grid).len() as f64 / secs
}

/// Base clock speed (hash units per second) using row base hashes over the timeline.
pub fn base_clock_hash_units_per_sec(run: &TimedRun) -> f64 {
    let secs = run.timing.total.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    let units: u128 = run.grid.rows.iter().map(|r| r.base_hash as u128).sum();
    (units as f64) / secs
}

/// Decode/select outputs along the timeline up to `depth` rows (years),
/// using the year series order as the base clock/timeline.
pub fn decode_by_timeline<'a>(run: &'a TimedRun, depth: usize) -> Vec<&'a PlankOutput> {
    select_depth(&run.grid, depth)
}

/// Await/collect all outputs across the grid.
pub fn await_outs(grid: &TimesPlankGrid) -> Vec<&PlankOutput> {
    grid.rows.iter().flat_map(|r| r.cells.iter()).collect()
}

/// Select outputs up to a decode depth (rows from start).
pub fn select_depth(grid: &TimesPlankGrid, depth: usize) -> Vec<&PlankOutput> {
    grid.rows
        .iter()
        .take(depth)
        .flat_map(|r| r.cells.iter())
        .collect()
}

/// Array field view of the grid: per-row vectors of output values.
pub fn array_field(grid: &TimesPlankGrid) -> Vec<Vec<i64>> {
    grid.rows
        .iter()
        .map(|r| r.cells.iter().map(|c| c.value).collect())
        .collect()
}

/// A calculation event for the hash-addressed event loop.
#[derive(Debug, Clone)]
pub struct CalcEvent {
    pub op: MathOp,
    pub a: i64,
    pub b: i64,
}

/// A node in the math tree structure, addressed/compared by hash.
#[derive(Debug, Clone)]
pub struct MathTreeNode {
    pub key_hash: u64,
    pub op: MathOp,
    pub a: i64,
    pub b: i64,
    pub row: GridRow,
    pub version: u64,
}

/// Hash-addressed math tree that is updated by the event loop.
#[derive(Debug, Clone, Default)]
pub struct MathTree {
    pub nodes: HashMap<u64, MathTreeNode>,
    pub event_log: Vec<String>,
}

/// Compute address key hash for a tree node (op + inputs define the address).
pub fn tree_key_hash(op: MathOp, a: i64, b: i64) -> u64 {
    let mut hasher = DefaultHasher::new();
    op.hash(&mut hasher);
    a.hash(&mut hasher);
    b.hash(&mut hasher);
    hasher.finish()
}

/// Process an event loop over calculations: for each event compute row/hash,
/// compare by address hash, and overwrite the tree node at that address when
/// the hash differs (or insert if new). This replicates the timeline/hash
/// addressing to update the tree structure of mathematics.
pub fn process_math_event_loop(tree: &mut MathTree, events: &[CalcEvent]) {
    for ev in events {
        let key = tree_key_hash(ev.op, ev.a, ev.b);
        let row = build_row_op(ev.op, ev.a, ev.b);
        let sym = match ev.op { MathOp::Mul => "x", MathOp::Div => "/" };
        match tree.nodes.get_mut(&key) {
            Some(node) => {
                if node.row.base_hash != row.base_hash || node.row.cells != row.cells {
                    node.version += 1;
                    node.row = row.clone();
                    node.op = ev.op;
                    node.a = ev.a;
                    node.b = ev.b;
                    tree.event_log.push(format!(
                        "overwrite key={} v={} {}{}{} hash={}",
                        key, node.version, ev.a, sym, ev.b, row.base_hash
                    ));
                } else {
                    tree.event_log.push(format!(
                        "same key={} {}{}{} hash={} (no overwrite)",
                        key, ev.a, sym, ev.b, row.base_hash
                    ));
                }
            }
            None => {
                tree.nodes.insert(key, MathTreeNode {
                    key_hash: key,
                    op: ev.op,
                    a: ev.a,
                    b: ev.b,
                    row: row.clone(),
                    version: 1,
                });
                tree.event_log.push(format!(
                    "insert key={} {}{}{} hash={}",
                    key, ev.a, sym, ev.b, row.base_hash
                ));
            }
        }
    }
}

/// Helper to build a mixed event stream (e.g. times and division tables).
pub fn build_table_events_mul(base: i64, n: i64) -> Vec<CalcEvent> {
    (1..=n.max(0)).map(|i| CalcEvent { op: MathOp::Mul, a: base, b: i }).collect()
}
pub fn build_table_events_div(dividend: i64, n: i64) -> Vec<CalcEvent> {
    (1..=n.max(0)).map(|i| CalcEvent { op: MathOp::Div, a: dividend, b: i }).collect()
}

/// Speculative array/matrix targeting matrix interpretations of fundamental calculations.
/// Represents drawn interpretations as a matrix of speculative candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeCell {
    pub row: usize,
    pub col: usize,
    pub value: f64,
    pub hash: u64,
    pub category: String,
    pub score: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SpeculativeMatrix {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<SpeculativeCell>,
}

fn hash_str_f64(parts: &[String], value: f64) -> u64 {
    let mut hasher = DefaultHasher::new();
    for p in parts {
        p.hash(&mut hasher);
    }
    value.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// Build a speculative matrix from grid outputs (fundamental calculations -> drawn matrix interpretation).
/// Each grid row becomes a matrix row; cells are speculative candidates per output.
pub fn speculative_matrix_from_grid(grid: &TimesPlankGrid, category: &str) -> SpeculativeMatrix {
    let rows = grid.rows.len();
    let cols = grid.rows.iter().map(|r| r.cells.len()).max().unwrap_or(0);
    let mut cells = Vec::new();
    for (i, row) in grid.rows.iter().enumerate() {
        for (j, out) in row.cells.iter().enumerate() {
            let value = out.value as f64;
            let cat = format!("{}:agent{}", category, out.agent_id);
            let h = hash_str_f64(&[cat.clone(), format!("{}:{}", i, j)], value);
            let score = (out.hash as f64) / (u64::MAX as f64);
            cells.push(SpeculativeCell { row: i, col: j, value, hash: h, category: cat, score });
        }
    }
    SpeculativeMatrix { rows, cols, cells }
}

/// Simple feature vector from a speculative cell for category/similarity search.
pub fn cell_feature_vector(cell: &SpeculativeCell) -> Vec<f64> {
    vec![cell.value, (cell.hash % 1_000_000) as f64 / 1_000_000.0, cell.score, cell.row as f64, cell.col as f64]
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

/// Image-search-like category sort: rank cells by (category match, similarity score) descending.
pub fn sort_by_category(matrix: &SpeculativeMatrix, query_category: &str, query_vector: Option<&[f64]>) -> Vec<SpeculativeCell> {
    let mut ranked = matrix.cells.clone();
    ranked.sort_by(|a, b| {
        let a_cat = if a.category.contains(query_category) { 1.0 } else { 0.0 };
        let b_cat = if b.category.contains(query_category) { 1.0 } else { 0.0 };
        let a_sim = query_vector.map(|q| cosine_similarity(&cell_feature_vector(a), q)).unwrap_or(a.score);
        let b_sim = query_vector.map(|q| cosine_similarity(&cell_feature_vector(b), q)).unwrap_or(b.score);
        let a_key = a_cat * 10.0 + a_sim;
        let b_key = b_cat * 10.0 + b_sim;
        b_key.partial_cmp(&a_key).unwrap_or(std::cmp::Ordering::Equal).then(a.hash.cmp(&b.hash))
    });
    ranked
}

/// Conceptual piece and math weights for perfected internal nodes.
#[derive(Debug, Clone)]
pub struct ConceptualPiece {
    pub id: u64,
    pub category: String,
    pub weights: Vec<f64>,
    pub evidence_hash: u64,
    pub version: u64,
}

#[derive(Debug, Clone, Default)]
pub struct MathWeightModel {
    pub nodes: HashMap<u64, ConceptualPiece>,
    pub learning_rate: f64,
    pub log: Vec<String>,
}

impl MathWeightModel {
    pub fn new(learning_rate: f64) -> Self {
        Self { nodes: HashMap::new(), learning_rate, log: Vec::new() }
    }

    /// Perfect internal node: create or update conceptual piece and adjust math weights
    /// based on observed cell/category evidence (hash-compared).
    pub fn perfect_node(&mut self, node_id: u64, category: &str, features: &[f64], evidence_hash: u64) {
        let lr = self.learning_rate;
        match self.nodes.get_mut(&node_id) {
            Some(piece) => {
                if piece.evidence_hash != evidence_hash || piece.category != category {
                    // update weights toward new features
                    let n = piece.weights.len().max(features.len());
                    piece.weights.resize(n, 0.0);
                    for (i, &f) in features.iter().enumerate() {
                        piece.weights[i] = piece.weights[i] * (1.0 - lr) + f * lr;
                    }
                    piece.category = category.to_string();
                    piece.evidence_hash = evidence_hash;
                    piece.version += 1;
                    self.log.push(format!("update node={} v={} cat={} hash={}", node_id, piece.version, category, evidence_hash));
                } else {
                    self.log.push(format!("same node={} hash={} (no update)", node_id, evidence_hash));
                }
            }
            None => {
                self.nodes.insert(node_id, ConceptualPiece {
                    id: node_id,
                    category: category.to_string(),
                    weights: features.to_vec(),
                    evidence_hash,
                    version: 1,
                });
                self.log.push(format!("insert node={} cat={} hash={}", node_id, category, evidence_hash));
            }
        }
    }
}

/// Integrate: from matrix + category sort, perfect model nodes and update weights.
pub fn refine_model_from_matrix(model: &mut MathWeightModel, matrix: &SpeculativeMatrix, query_category: &str) {
    let ranked = sort_by_category(matrix, query_category, None);
    for (i, cell) in ranked.iter().take(matrix.cells.len()).enumerate() {
        let node_id = hash_str_f64(&[cell.category.clone(), i.to_string()], cell.value);
        let feats = cell_feature_vector(cell);
        model.perfect_node(node_id, &cell.category, &feats, cell.hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_digit_transform_example_12x8() {
        let outs = times_plank_multiply(12, 8);
        assert!(outs.iter().any(|o| o.agent_id == 0 && o.value == 816));
    }

    #[test]
    fn standard_included() {
        let outs = times_plank_multiply(13, 8);
        assert!(outs.iter().any(|o| o.value == 104));
    }

    #[test]
    fn agent_sorting_by_hash() {
        let mut outs = times_plank_multiply(13, 8);
        sort_agents_by_score(&mut outs);
        for w in outs.windows(2) {
            assert!(w[0].hash <= w[1].hash);
        }
    }

    #[test]
    fn grid_fields() {
        let grid = build_grid(&[(12, 8), (13, 8)]);
        assert_eq!(grid.rows.len(), 2);
        assert!(!grid.log.is_empty());
        assert!(grid.num_positive > 0);
        assert_eq!(await_outs(&grid).len(), select_depth(&grid, 2).len());
        assert_eq!(array_field(&grid).len(), 2);
    }

    #[test]
    fn timed_year_series_and_clock() {
        let years = vec![1, 2, 10, 1000];
        let run = run_year_series(8, &years);
        assert_eq!(run.grid.rows.len(), years.len());
        assert_eq!(run.timing.per_row.len(), years.len());
        assert!(run.timing.total >= Duration::from_nanos(0));
        assert!(!decode_by_timeline(&run, 2).is_empty());
        let _clock = base_clock_outputs_per_sec(&run);
        let _hash_clock = base_clock_hash_units_per_sec(&run);
    }

    #[test]
    fn division_outputs_and_table() {
        let outs = times_plank_divide(84, 4);
        assert!(outs.iter().any(|o| o.agent_id == 1 && o.value == 21));
        let grid = division_table(100, 10);
        assert_eq!(grid.rows.len(), 10);
        assert!(!await_outs(&grid).is_empty());
    }

    #[test]
    fn division_series_timed() {
        let run = run_division_series(8000, &[1, 2, 4, 8, 10, 100, 1000]);
        assert_eq!(run.grid.rows.len(), 7);
        assert!(run.timing.total >= Duration::from_nanos(0));
    }

    #[test]
    fn math_tree_event_loop_overwrite_by_hash() {
        let mut tree = MathTree::default();
        let events = vec![
            CalcEvent { op: MathOp::Mul, a: 12, b: 8 },
            CalcEvent { op: MathOp::Div, a: 84, b: 4 },
            CalcEvent { op: MathOp::Mul, a: 12, b: 8 }, // same address -> compare hash, no overwrite
        ];
        process_math_event_loop(&mut tree, &events);
        assert_eq!(tree.nodes.len(), 2);
        assert!(tree.event_log.iter().any(|l| l.contains("insert")));
        assert!(tree.event_log.iter().any(|l| l.contains("same") || l.contains("no overwrite")));

        // force different outcome is not possible with pure fn; but we can process a different calc at same logical slot by new event stream including update path
        let mut tree2 = MathTree::default();
        process_math_event_loop(&mut tree2, &build_table_events_mul(8, 4));
        process_math_event_loop(&mut tree2, &build_table_events_div(64, 4));
        assert!(tree2.nodes.len() >= 8);
    }
}
