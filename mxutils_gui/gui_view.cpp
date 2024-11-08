#include "gui_view.h"


#define LabelDefaultStyle (          \
    "QLabel {"                       \
    "    font-size: 20px;"           \
    "    font-weight: bold;"         \
    "    color: #14FF39;"            \
    "    background-color: #646464;" \
    "    border: 2px solid black;"   \
    "    border-radius: 5px;"        \
    "    padding: 5px;"              \
    "    text-align: center;"        \
    "}")

#define PushButtonEnableStyle (      \
    "QPushButton {"                  \
    "    font-size: 19px;"           \
    "    font-weight: bold;"         \
    "    color: #14FF39;"            \
    "    background-color: #646464;" \
    "    padding: 5px;"              \
    "    text-align: center;"        \
    "}")

#define PushButtonDisableStyle (     \
    "QPushButton {"                  \
    "    font-size: 19px;"           \
    "    font-weight: bold;"         \
    "    color: #646464;"            \
    "    background-color: #646464;" \
    "    padding: 5px;"              \
    "    text-align: center;"        \
    "}")

MxQt::MxQt(int &argc, char *argv[]) : app(argc, argv)
{
    int width_offset = 0;

    num_screens = app.screens().size();

    for (int i = 0; i < num_screens; i++)
    {
        QScreen *qscreen = app.screens().at(i);
        DisplayScreen *screen = new DisplayScreen(nullptr, qscreen);
        int w = app.screens().at(i)->geometry().width();
        int h = app.screens().at(i)->geometry().height();
        screen->setGeometry(width_offset, 0, w, h);
        width_offset += app.screens().at(i)->geometry().width();
        screens.push_back(screen);
    }
}

int MxQt::Run()
{
    for (int i = 0; i < num_screens; i++)
        screens[i]->show();

    return app.exec();
}

MxQt::~MxQt()
{
    for (uint32_t i = 0; i < screens.size(); i++)
        delete screens[i];
}

DisplayScreen::DisplayScreen() : QWidget() // default constructor
{
    this->running_ = true;
}

DisplayScreen::DisplayScreen(QWidget *parent = nullptr, QScreen *qscreen = nullptr) : QWidget(parent)
{
    this->running_ = true;
    this->w_ = qscreen->geometry().width();
    this->h_ = qscreen->geometry().height();
}

DisplayScreen::~DisplayScreen()
{
    this->running_ = false;
    for (uint32_t idx = 0; idx < this->num_viewers_; idx++)
    {
        QObject *object = viewers_[idx];
        FrameViewer *viewer = (FrameViewer *)object;
        delete viewer;
    }
}

// TODO: make it a function directly in FrameViewer
void DisplayScreen::SetDisplayFrame(int viewer_id, cv::Mat *frame, float fps)
{
    QObject *object = viewers_[viewer_id];

    FrameViewer *viewer = (FrameViewer *)object;
    viewer->UpdateFrame(frame);
    viewer->UpdateFPS(fps);
}

void DisplayScreen::SetDisplayFrame(int viewer_id, cv::Mat *frame)
{
    QObject *object = viewers_[viewer_id];

    FrameViewer *viewer = (FrameViewer *)object;
    viewer->UpdateFrame(frame);
    viewer->HideFPS();
    viewer->HideChannelName();
}

void DisplayScreen::SetDisplayFrame(int viewer_id, cv::Mat frame)
{
    SetDisplayFrame(viewer_id, &frame);
}

cv::Mat *DisplayScreen::GetDisplayFrameBuf(int viewer_id)
{
    QObject *object = viewers_[viewer_id];

    FrameViewer *viewer = (FrameViewer *)object;
    cv::Mat *display_frame = viewer->display_frame_list_.at(viewer->display_frame_idx_);
    viewer->display_frame_idx_ = (viewer->display_frame_idx_ + 1) % viewer->display_frame_list_.size();
    return display_frame;
}

uint32_t DisplayScreen::width()
{
    return this->w_;
}

uint32_t DisplayScreen::height()
{
    return this->h_;
}

uint32_t DisplayScreen::GetViewerWidth(int viewer_id)
{
    QObject *object = viewers_[viewer_id];

    FrameViewer *viewer = (FrameViewer *)object;
    return viewer->width();
}

uint32_t DisplayScreen::GetViewerHeight(int viewer_id)
{
    QObject *object = viewers_[viewer_id];

    FrameViewer *viewer = (FrameViewer *)object;
    return viewer->height();
}

void DisplayScreen::SetSquareLayout(int num_channels, bool fullscreen)
{
    static int idx = 0;
    bool create_exit_button;
    int screen_width;
    int screen_height;
    if(fullscreen){
        create_exit_button = false;
        screen_width = this->width();
        screen_height = this->height();
    }
    else{
        create_exit_button = true;
        this->setFixedSize(1280, 720);
        screen_width = 1280;
        screen_height = 720;
    }
    vector<ViewerGeometry> viewer_geometry;

    {
        // NxN layout
        int mode = ceil(sqrt(num_channels));

        int geo_w, geo_h;
        geo_w = int((screen_width) / mode);
        geo_w &= (~0x1f); // sws_scale needs width to be 32x
        geo_h = int((screen_height) / mode);
        geo_h = (geo_w * 9) / 16;

        viewer_geometry.clear();

        for (int i = 0; i < num_channels; i++)
        {
            int x = (i % mode) * geo_w;
            int y = ((i / mode) * (geo_h));
            ViewerGeometry cg = {x, y, geo_w, geo_h};
            viewer_geometry.push_back(cg);
        }

        this->viewer_geometry_ = viewer_geometry;
    }

    // init and set all necessary components in each widget

    for (int i = 0; i < num_channels; i++)
    {
        ViewerGeometry &cg = viewer_geometry.at(i);

        FrameViewer *viewer = new FrameViewer(this);

        viewer->SetGeometry(cg.x, cg.y, cg.w, cg.h);
        viewer->SetIdx(idx++); // set id for parent to distinguish
        this->AddViewer(viewer);
    }

    if (!create_exit_button)
    {
        QPushButton *exitButton = new QPushButton("Exit", this);
        exitButton->setGeometry(screen_width - 60, 0, 60, 25); // FIXME
        QObject::connect(exitButton, &QPushButton::clicked, &QCoreApplication::quit);
        create_exit_button = true;
    }

    if(fullscreen)
    this->setWindowState(Qt::WindowFullScreen);
}

void DisplayScreen::AddViewer(QWidget *viewer)
{
    viewers_.push_back(viewer);
    num_viewers_ = viewers_.size();
}

uint32_t DisplayScreen::NumViewers()
{
    return num_viewers_;
}

FrameViewer::FrameViewer(DisplayScreen *parent = nullptr) : QWidget(parent)
{
    connect(this, SIGNAL(signal_UpdateFrame(cv::Mat *)), this, SLOT(slot_UpdateFrame(cv::Mat *)));
    connect(this, SIGNAL(signal_UpdateFPS(float)), this, SLOT(slot_UpdateFPS(float)));

    this->running_ = true;
    this->display_frame_idx_ = 0;

    frame_ = new QLabel(this);

    int count = 0;
    int interval = 35; // FIXME

    name_ = new QLabel(frame_);
    name_->setStyleSheet(LabelDefaultStyle);
    name_->move(0, count * interval);

    fps_ = new QLabel("FPS = ", frame_);
    fps_->setStyleSheet(LabelDefaultStyle);
    fps_->move(name_->width() - 3, count * interval);
    count++;
    // Set up the layout
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(frame_);
    layout->setSpacing(0);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);
}

FrameViewer::~FrameViewer()
{
    this->running_ = false;
    int size = display_frame_list_.size();
    for (int i = 0; i < size; i++)
        delete display_frame_list_[i];
}

uint32_t FrameViewer::width()
{
    return this->w_;
}

uint32_t FrameViewer::height()
{
    return this->h_;
}

void FrameViewer::SetGeometry(int x, int y, int w, int h)
{
    x_ = x;
    y_ = y;
    w_ = w;
    h_ = h;
    this->setGeometry(x, y, w, h);
    // Set up display frame buffer
    for (int i = 0; i < 60 /* FIXME */; i++)
    {
        cv::Mat *display_frame = new cv::Mat(h_, w_, CV_8UC3);
        this->display_frame_list_.push_back(display_frame);
    }
}

void FrameViewer::SetIdx(int idx)
{
    this->idx_ = idx;
    this->setObjectName(QString::number(idx_));
    name_->setText("CH" + QString::number(idx_)); // zero-index
    name_->adjustSize();
}

void FrameViewer::UpdateFrame(cv::Mat *frame)
{
    emit signal_UpdateFrame(frame);
}

void FrameViewer::slot_UpdateFrame(cv::Mat *frame)
{
    QImage img((*frame).data, (*frame).cols, (*frame).rows, (*frame).step, QImage::Format_RGB888);
    // Set the QImage as the pixmap for the QLabel
    QPixmap pixmap = QPixmap::fromImage(img);
    if (pixmap.isNull()) {
        throw std::runtime_error("QtUtil error: Failed to load image.");
    // Handle the error accordingly
    }
    frame_->setPixmap(pixmap.scaled(frame_->size(),Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
}

void FrameViewer::UpdateFPS(float fps)
{
    emit signal_UpdateFPS(fps);
}

void FrameViewer::slot_UpdateFPS(float fps)
{
    if (fps == .0)
        return;
    fps_->move(name_->width() - 3, 0);
    fps_->setText("FPS = " + QString::number(fps, 'f', 1));
    fps_->adjustSize();
}

void FrameViewer::HideFPS()
{
    this->fps_->hide();
}

void FrameViewer::HideChannelName()
{
    this->name_->hide();
}
