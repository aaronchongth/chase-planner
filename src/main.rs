use bevy::prelude::*;
use bevy::math::IVec2;
use noise::{NoiseFn, Perlin};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;

// --- Constants ---
const GRID_WIDTH: usize = 100;
const GRID_HEIGHT: usize = 100;
const BLOCK_SIZE: f32 = 6.0; // Size of each block in pixels
const GAP_SIZE: f32 = 1.0; // Gap between blocks

const NAVIGABLE_COLOR: Color = Color::GRAY;
const NON_NAVIGABLE_COLOR: Color = Color::WHITE;
const PATH_COLOR: Color = Color::rgba(1.0, 0.4, 0.4, 0.8); // Pale red

// --- Path Generation Constants ---
const PATH_LENGTH: usize = 100;

// --- Simulation Constants ---
const RUNNER_STEP_INTERVAL: f32 = 0.5; // seconds

// Calculate window dimensions based on grid size
const WINDOW_WIDTH: f32 = (BLOCK_SIZE + GAP_SIZE) * GRID_WIDTH as f32;
const WINDOW_HEIGHT: f32 = (BLOCK_SIZE + GAP_SIZE) * GRID_HEIGHT as f32;

// --- Noise Generation Constants ---
const NOISE_SCALE: f64 = 0.05; // Smaller values -> larger, smoother features.
const NAVIGABILITY_THRESHOLD: f64 = 0.1; // Range: -1.0 to 1.0. Higher values -> more navigable space.

// --- State ---
#[derive(Debug, Clone, Eq, PartialEq, Hash, States, Default)]
enum SimulationState {
    #[default]
    Paused,
    Running,
}

// --- Components ---

/// A component to identify a grid cell entity and its position in the grid.
#[derive(Component)]
struct GridCell {
    x: usize,
    y: usize,
}

#[derive(Component)]
struct RandomizeButton;

#[derive(Component)]
struct Runner;

#[derive(Component)]
struct PlayPauseButton;

#[derive(Component)]
struct PlayPauseButtonText;

// --- Resources ---

/// A resource to hold the runner's path as a sequence of grid coordinates.
#[derive(Resource, Default)]
struct RunnerPath(Vec<IVec2>);

#[derive(Resource)]
struct SimulationTimer(Timer);

/// A resource to hold the state of the grid.
#[derive(Resource)]
struct Grid {
    /// `true` for navigable (grey), `false` for non-navigable (white).
    cells: Vec<bool>,
    width: usize,
    height: usize,
}

impl Grid {
    /// Creates a new grid with all cells set to non-navigable.
    fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![false; width * height],
            width,
            height,
        }
    }

    /// Generates a map using Perlin noise to create clustered patterns.
    fn randomize(&mut self) {
        // Use a different seed each time for variety
        let seed = rand::thread_rng().gen();
        let perlin = Perlin::new(seed);

        for y in 0..self.height {
            for x in 0..self.width {
                let index = self.get_index(x, y);
                // Scale coordinates for the noise function
                let nx = x as f64 * NOISE_SCALE;
                let ny = y as f64 * NOISE_SCALE;

                // Get the noise value, which is between -1.0 and 1.0
                let value = perlin.get([nx, ny]);

                // If the value is below a threshold, the cell is navigable.
                self.cells[index] = value < NAVIGABILITY_THRESHOLD;
            }
        }
    }

    /// Gets the 1D vector index for a cell at (x, y).
    fn get_index(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Grid World".into(),
                resolution: (WINDOW_WIDTH, WINDOW_HEIGHT).into(),
                ..default()
            }),
            ..default()
        }))
        .init_state::<SimulationState>()
        .insert_resource(SimulationTimer(Timer::from_seconds(
            RUNNER_STEP_INTERVAL,
            TimerMode::Repeating,
        )))
        .init_resource::<RunnerPath>()
        .insert_resource(Grid::new(GRID_WIDTH, GRID_HEIGHT))
        .add_systems(Startup, (setup_camera, setup_grid, setup_ui))
        .add_systems(
            Update,
            (
                randomize_button_system,
                play_button_system,
                update_play_button_text.run_if(state_changed::<SimulationState>),
                runner_simulation_system.run_if(in_state(SimulationState::Running)),
                update_runner_transform.run_if(resource_changed::<RunnerPath>),
                draw_path_system,
                // This system runs only when the Grid resource has changed.
                // This is an optimization to avoid redrawing the grid every frame.
                color_grid_cells.run_if(resource_changed::<Grid>),
            ),
        )
        .run();
}

/// Sets up the 2D camera.
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

/// Spawns the grid of sprites, one for each cell.
fn setup_grid(
    mut commands: Commands,
    mut grid: ResMut<Grid>,
    mut path: ResMut<RunnerPath>,
) {
    // Randomize the grid on startup for an initial pattern
    grid.randomize();
    path.0 = generate_path(&grid, PATH_LENGTH);

    // Spawn the runner at the start of the path
    if let Some(start_pos) = path.0.first() {
        commands.spawn((
            SpriteBundle {
                sprite: Sprite {
                    color: Color::ORANGE_RED,
                    custom_size: Some(Vec2::splat(BLOCK_SIZE * 1.5)), // Make it slightly larger
                    ..default()
                },
                // The runner's Z-value is 2.0 to ensure it's drawn on top of the path.
                transform: Transform::from_translation(grid_to_world(*start_pos, 2.0)),
                ..default()
            },
            Runner,
        ));
    }

    let total_width = GRID_WIDTH as f32 * (BLOCK_SIZE + GAP_SIZE);
    let total_height = GRID_HEIGHT as f32 * (BLOCK_SIZE + GAP_SIZE);

    // Calculate the starting position to center the grid
    let start_x = -total_width / 2.0 + BLOCK_SIZE / 2.0;
    let start_y = -total_height / 2.0 + BLOCK_SIZE / 2.0;

    for y in 0..GRID_HEIGHT {
        for x in 0..GRID_WIDTH {
            let x_pos = start_x + x as f32 * (BLOCK_SIZE + GAP_SIZE);
            let y_pos = start_y + y as f32 * (BLOCK_SIZE + GAP_SIZE);

            commands.spawn((
                SpriteBundle {
                    sprite: Sprite {
                        // Color is set by the `color_grid_cells` system
                        custom_size: Some(Vec2::new(BLOCK_SIZE, BLOCK_SIZE)),
                        ..default()
                    },
                    transform: Transform::from_xyz(x_pos, y_pos, 0.0),
                    ..default()
                },
                // Add our custom component to link this sprite to a grid cell
                GridCell { x, y },
            ));
        }
    }
}

/// Updates the color of each grid cell sprite based on the `Grid` resource.
fn color_grid_cells(grid: Res<Grid>, mut query: Query<(&GridCell, &mut Sprite)>) {
    for (cell, mut sprite) in query.iter_mut() {
        let is_navigable = grid.cells[grid.get_index(cell.x, cell.y)];
        sprite.color = if is_navigable {
            NAVIGABLE_COLOR
        } else {
            NON_NAVIGABLE_COLOR
        };
    }
}

/// Spawns the "Randomize" button UI.
fn setup_ui(mut commands: Commands) {
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.0),
                flex_direction: FlexDirection::Row,
                height: Val::Percent(100.0),
                justify_content: JustifyContent::FlexStart, // Align to top
                align_items: AlignItems::FlexStart,         // Align to left
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            // --- Randomize Button ---
            parent
                .spawn((
                    ButtonBundle {
                        style: Style {
                            width: Val::Px(150.0),
                            height: Val::Px(50.0),
                            margin: UiRect::all(Val::Px(10.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.15, 0.15, 0.15).into(),
                        ..default()
                    },
                    RandomizeButton,
                ))
                .with_children(|parent| {
                    parent.spawn(TextBundle::from_section(
                        "Randomize",
                        TextStyle {
                            font_size: 20.0,
                            color: Color::WHITE,
                            ..default()
                        },
                    ));
                });

            // --- Play/Pause Button ---
            parent
                .spawn((
                    ButtonBundle {
                        style: Style {
                            width: Val::Px(150.0),
                            height: Val::Px(50.0),
                            margin: UiRect::all(Val::Px(10.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        background_color: Color::rgb(0.15, 0.15, 0.85).into(), // Blueish
                        ..default()
                    },
                    PlayPauseButton,
                ))
                .with_children(|parent| {
                    parent.spawn((
                        TextBundle::from_section(
                            "Play",
                            TextStyle {
                                font_size: 20.0,
                                color: Color::WHITE,
                                ..default()
                            },
                        ),
                        PlayPauseButtonText,
                    ));
                });
        });
}

/// Handles button interactions to randomize the grid.
fn randomize_button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<RandomizeButton>),
    >,
    mut grid: ResMut<Grid>,
    mut path: ResMut<RunnerPath>,
    mut next_state: ResMut<NextState<SimulationState>>,
) {
    for (interaction, mut color) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = Color::rgb(0.35, 0.75, 0.35).into(); // Green when pressed
                grid.randomize();
                path.0 = generate_path(&grid, PATH_LENGTH);
                next_state.set(SimulationState::Paused);
            }
            Interaction::Hovered => {
                *color = Color::rgb(0.25, 0.25, 0.25).into(); // Lighter grey on hover
            }
            Interaction::None => {
                *color = Color::rgb(0.15, 0.15, 0.15).into(); // Default dark grey
            }
        }
    }
}

/// Toggles the simulation state when the play/pause button is clicked.
fn play_button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<PlayPauseButton>),
    >,
    current_state: Res<State<SimulationState>>,
    mut next_state: ResMut<NextState<SimulationState>>,
) {
    for (interaction, mut color) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => match current_state.get() {
                SimulationState::Paused => next_state.set(SimulationState::Running),
                SimulationState::Running => next_state.set(SimulationState::Paused),
            },
            Interaction::Hovered => {
                *color = Color::rgb(0.25, 0.25, 0.95).into();
            }
            Interaction::None => {
                *color = Color::rgb(0.15, 0.15, 0.85).into();
            }
        }
    }
}

/// Updates the text of the play/pause button based on the current simulation state.
fn update_play_button_text(
    mut text_query: Query<&mut Text, With<PlayPauseButtonText>>,
    current_state: Res<State<SimulationState>>,
) {
    if let Ok(mut text) = text_query.get_single_mut() {
        text.sections[0].value = match current_state.get() {
            SimulationState::Paused => "Play".to_string(),
            SimulationState::Running => "Pause".to_string(),
        };
    }
}

/// Generates a random path (walk) on navigable tiles.
fn generate_path(grid: &Grid, length: usize) -> Vec<IVec2> {
    let mut rng = rand::thread_rng();

    // Find all navigable tiles to use as potential starting points.
    let navigable_tiles: Vec<IVec2> = (0..grid.height)
        .flat_map(|y| (0..grid.width).map(move |x| IVec2::new(x as i32, y as i32)))
        .filter(|pos| grid.cells[grid.get_index(pos.x as usize, pos.y as usize)])
        .collect();

    if navigable_tiles.is_empty() {
        return vec![]; // No place to walk.
    }

    let mut path = Vec::with_capacity(length);
    let mut visited = HashSet::new();

    // 1. Find a random starting position and add it to the path.
    let start_pos = *navigable_tiles.choose(&mut rng).unwrap();
    path.push(start_pos);
    visited.insert(start_pos);

    let mut current_pos = start_pos;

    // Define 8-directional neighbors (including diagonals).
    const NEIGHBORS: [IVec2; 8] = [
        IVec2::new(-1, 1), IVec2::new(0, 1), IVec2::new(1, 1),
        IVec2::new(-1, 0),                  IVec2::new(1, 0),
        IVec2::new(-1, -1), IVec2::new(0, -1), IVec2::new(1, -1),
    ];

    // 2. Perform the first step randomly to establish an initial direction.
    let first_step_neighbors: Vec<IVec2> = NEIGHBORS.iter()
        .map(|&offset| current_pos + offset)
        .filter(|&p| {
            p.x >= 0 && p.x < grid.width as i32 && p.y >= 0 && p.y < grid.height as i32
            && grid.cells[grid.get_index(p.x as usize, p.y as usize)]
        })
        .collect();

    let mut last_direction = if let Some(&next_pos) = first_step_neighbors.choose(&mut rng) {
        path.push(next_pos);
        visited.insert(next_pos);
        current_pos = next_pos;
        next_pos - start_pos
    } else {
        return path; // Nowhere to go from the start.
    };

    // 3. Perform a biased random walk for the rest of the path.
    for _ in 2..length {
        let weighted_neighbors: Vec<(IVec2, f32)> = NEIGHBORS.iter()
            .map(|&offset| current_pos + offset)
            .filter(|&p| {
                // Check if the neighbor is within grid bounds.
                p.x >= 0 && p.x < grid.width as i32 && p.y >= 0 && p.y < grid.height as i32
                // Check if the tile is navigable.
                && grid.cells[grid.get_index(p.x as usize, p.y as usize)]
                // Check that we haven't visited this tile yet in this path.
                && !visited.contains(&p)
            })
            .map(|p| {
                let move_dir = (p - current_pos).as_vec2().normalize_or_zero();
                let last_dir = last_direction.as_vec2().normalize_or_zero();
                let dot_product = move_dir.dot(last_dir);
                // This formula creates a weight. Forward moves (dot ≈ 1) are heavily favored.
                // Sideways moves (dot ≈ 0) are neutral. Backward moves (dot ≈ -1) are heavily disfavored.
                let weight = (1.0f32 + dot_product).powi(4);
                (p, weight)
            })
            .collect();

        if let Ok(choice) = weighted_neighbors.choose_weighted(&mut rng, |item: &(IVec2, f32)| item.1) {
            let next_pos = choice.0; // The chosen IVec2 position
            last_direction = next_pos - current_pos;
            current_pos = next_pos;
            path.push(current_pos);
            visited.insert(current_pos);
        } else {
            // If we hit a dead end, stop generating the path.
            break;
        }
    }

    path
}

/// Extends the given path by one step using a biased random walk.
fn extend_path(grid: &Grid, path: &mut Vec<IVec2>) {
    if path.len() < 2 {
        // Not enough history to determine a direction, so we can't proceed.
        return;
    }

    let mut rng = rand::thread_rng();
    let current_pos = *path.last().unwrap();
    let prev_pos = path[path.len() - 2];
    let last_direction = current_pos - prev_pos;

    const NEIGHBORS: [IVec2; 8] = [
        IVec2::new(-1, 1), IVec2::new(0, 1), IVec2::new(1, 1),
        IVec2::new(-1, 0),                  IVec2::new(1, 0),
        IVec2::new(-1, -1), IVec2::new(0, -1), IVec2::new(1, -1),
    ];

    // Create a set of already used path points for quick lookups.
    let visited: HashSet<IVec2> = path.iter().cloned().collect();

    let weighted_neighbors: Vec<(IVec2, f32)> = NEIGHBORS.iter()
        .map(|&offset| current_pos + offset)
        .filter(|&p| {
            p.x >= 0 && p.x < grid.width as i32 && p.y >= 0 && p.y < grid.height as i32
            && grid.cells[grid.get_index(p.x as usize, p.y as usize)]
            && !visited.contains(&p)
        })
        .map(|p| {
            let move_dir = (p - current_pos).as_vec2().normalize_or_zero();
            let last_dir = last_direction.as_vec2().normalize_or_zero();
            let dot_product = move_dir.dot(last_dir);
            let weight = (1.0f32 + dot_product).powi(4);
            (p, weight)
        })
        .collect();

    if let Ok(choice) = weighted_neighbors.choose_weighted(&mut rng, |item| item.1) {
        path.push(choice.0);
    } else {
        // If we hit a dead end, try a non-biased random walk to escape.
        let unweighted_neighbors: Vec<_> = NEIGHBORS.iter()
            .map(|&offset| current_pos + offset)
            .filter(|&p| {
                p.x >= 0 && p.x < grid.width as i32 && p.y >= 0 && p.y < grid.height as i32
                && grid.cells[grid.get_index(p.x as usize, p.y as usize)]
                && !visited.contains(&p)
            })
            .collect();
        if let Some(fallback_choice) = unweighted_neighbors.choose(&mut rng) {
            path.push(*fallback_choice);
        }
        // If still no path, the runner is truly stuck. The path will shorten on the next step.
    }
}

/// Converts grid coordinates (e.g., 0,0) to world coordinates (e.g., -350.0, -350.0).
fn grid_to_world(grid_pos: IVec2, z: f32) -> Vec3 {
    let total_width = GRID_WIDTH as f32 * (BLOCK_SIZE + GAP_SIZE);
    let total_height = GRID_HEIGHT as f32 * (BLOCK_SIZE + GAP_SIZE);
    let start_x = -total_width / 2.0 + BLOCK_SIZE / 2.0;
    let start_y = -total_height / 2.0 + BLOCK_SIZE / 2.0;

    let x_pos = start_x + grid_pos.x as f32 * (BLOCK_SIZE + GAP_SIZE);
    let y_pos = start_y + grid_pos.y as f32 * (BLOCK_SIZE + GAP_SIZE);

    Vec3::new(x_pos, y_pos, z)
}

/// Draws the runner's path using gizmos.
fn draw_path_system(path: Res<RunnerPath>, mut gizmos: Gizmos) {
    // We need at least two points to draw a line.
    if path.0.len() < 2 {
        return;
    }

    // Create a polyline from the waypoints.
    // Note: Bevy's Gizmos do not support dashed lines out of the box. This will be a solid line.
    // The path's Z-value is 1.0 to be drawn on top of the grid but below the runner.
    let world_points = path.0.iter().map(|&p| grid_to_world(p, 1.0));
    gizmos.linestrip(world_points, PATH_COLOR);
}

/// In the running state, advances the simulation every time the timer finishes.
fn runner_simulation_system(
    time: Res<Time>,
    mut timer: ResMut<SimulationTimer>,
    mut path: ResMut<RunnerPath>,
    grid: Res<Grid>,
) {
    if timer.0.tick(time.delta()).just_finished() {
        if !path.0.is_empty() {
            path.0.remove(0);
            extend_path(&grid, &mut path.0);
        }
    }
}

/// Keeps the runner's visual position synced with the first waypoint in the path.
fn update_runner_transform(
    mut runner_query: Query<&mut Transform, With<Runner>>,
    path: Res<RunnerPath>,
) {
    if let (Ok(mut transform), Some(target_pos)) = (runner_query.get_single_mut(), path.0.first()) {
        transform.translation = grid_to_world(*target_pos, 2.0); // Z=2.0 to be on top
    }
}
