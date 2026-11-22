import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const predict = {
    tabular: (data: any, diseaseType?: string) => api.post('/predict/tabular', { ...data, disease_type: diseaseType || 'Heart Disease' }),
    uploadCsv: (formData: FormData) => api.post('/predict/upload_csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    }),
    getUniqueDiseases: () => api.get<string[]>('/predict/datasets/unique-diseases'),
};

export const chat = {
    start: (disease: string) => api.post('/chat/start', null, { params: { disease_context: disease } }),
    message: (sessionId: number, content: string) => api.post(`/chat/${sessionId}/message`, null, { params: { content } }),
    getHistory: (sessionId: number) => api.get(`/chat/${sessionId}/history`),
};

export const dashboard = {
    getAdminStats: () => api.get('/predict/dashboard/admin-stats'),
    listAdminUploads: () => api.get<{ filenames: string[] }>('/predict/dashboard/admin-uploads'),
    getUserStats: () => api.get('/predict/dashboard/user-stats'),
    getHistory: () => api.get('/predict/predict/history'), // Note: router prefix is /predict, check nesting
    getWorkspaceStats: () => api.get('/predict/dashboard/admin-stats'),
    listWorkspaceUploads: () => api.get<{ filenames: string[] }>('/predict/dashboard/admin-uploads'),
};

export const experiments = {
    getResults: () => api.get('/predict/experimental-results'),
    regenerate: () => api.post('/predict/experimental-results/regenerate'),
};

export default api;
