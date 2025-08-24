use bevy::prelude::*;
use noise::{NoiseFn, Perlin};
use rand::Rng;

// --- Constants ---
const GRID_WIDTH: usize = 100;
const GRID_HEIGHT: usize = 100;
const BLOCK_SIZE: f32 = 6.0; // Size of each block in pixels
const GAP_SIZE: f32 = 1.0; // Gap between blocks

const NAVIGABLE_COLOR: Color = Color::GRAY;
const NON_NAVIGABLE_COLOR: Color = Color::WHITE;

// Calculate window dimensions based on grid size
const WINDOW_WIDTH: f32 = (BLOCK_SIZE + GAP_SIZE) * GRID_WIDTH as f32;
const WINDOW_HEIGHT: f32 = (BLOCK_SIZE + GAP_SIZE) * GRID_HEIGHT as f32;

// --- Noise Generation Constants ---
const NOISE_SCALE: f64 = 0.05; // Smaller values -> larger, smoother features.
const NAVIGABILITY_THRESHOLD: f64 = 0.1; // Range: -1.0 to 1.0. Higher values -> more navigable space.

// --- Components ---

/// A component to identify a grid cell entity and its position in the grid.
#[derive(Component)]
struct GridCell {
    x: usize,
    y: usize,
}

// --- Resources ---//

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
        .insert_resource(Grid::new(GRID_WIDTH, GRID_HEIGHT))
        .add_systems(Startup, (setup_camera, setup_grid, setup_ui))
        .add_systems(
            Update,
            (
                button_system,
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
fn setup_grid(mut commands: Commands, mut grid: ResMut<Grid>) {
    // Randomize the grid on startup for an initial pattern
    grid.randomize();

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
                height: Val::Percent(100.0),
                justify_content: JustifyContent::FlexStart, // Align to top
                align_items: AlignItems::FlexStart,         // Align to left
                ..default()
            },
            ..default()
        })
        .with_children(|parent| {
            parent
                .spawn(ButtonBundle {
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
                })
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
        });
}

/// Handles button interactions to randomize the grid.
fn button_system(
    mut interaction_query: Query<
        (&Interaction, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
    mut grid: ResMut<Grid>,
) {
    for (interaction, mut color) in &mut interaction_query {
        match *interaction {
            Interaction::Pressed => {
                *color = Color::rgb(0.35, 0.75, 0.35).into(); // Green when pressed
                grid.randomize();
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
