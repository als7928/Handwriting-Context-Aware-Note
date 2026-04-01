pipeline {
    agent { label 'docker' } 

    tools {
        'org.jenkinsci.plugins.docker.commons.tools.DockerTool' 'jenkins-docker'
    }

    environment {
        HARBOR_URL = 'amdp-registry.skala-ai.com'
        HARBOR_PROJECT = 'skala26a-ai2'
        
        HARBOR_CREDS = credentials('harbor-robot-account') 
        
        BACKEND_IMAGE = 'sk047-myservice-backend'
        BACKEND_VER = '1.0.4'
        FRONTEND_IMAGE = 'sk047-myservice-frontend'
        FRONTEND_VER = '1.0.1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            steps {
                script {
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        echo ">>> Building Images..."
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} -f backend/Dockerfile-backend ./backend"
                        sh "docker build -t ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} -f frontend/Dockerfile-frontend ./frontend"
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    echo '>>> Stage 4: Logging in to Harbor and Pushing'
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        sh "echo ${HARBOR_CREDS_PSW} | docker login ${HARBOR_URL} -u '${HARBOR_CREDS_USR}' --password-stdin"
                        
                        // 2. Push Images
                        echo "Pushing Backend..."
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER}"
                        
                        echo "Pushing Frontend..."
                        sh "docker push ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER}"
                        
                        // 3. Logout
                        sh "docker logout ${HARBOR_URL}"
                    }
                }
            }
        }
    }

    post {
        always {
            script {
                try {
                    def dockerHome = tool name: 'jenkins-docker', type: 'org.jenkinsci.plugins.docker.commons.tools.DockerTool'
                    withEnv(["PATH+DOCKER=${dockerHome}/bin"]) {
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${BACKEND_IMAGE}:${BACKEND_VER} || true"
                        sh "docker rmi ${HARBOR_URL}/${HARBOR_PROJECT}/${FRONTEND_IMAGE}:${FRONTEND_VER} || true"
                    }
                } catch (e) {
                    echo "Cleanup cleanup skipped."
                }
            }
        }
    }
}